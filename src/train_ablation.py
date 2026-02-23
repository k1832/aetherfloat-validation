import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
from aether_sim import float_to_af_tensor

class AetherSim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, chunk_size):
        ctx.bits = bits
        ctx.chunk_size = chunk_size
        x_cloned = x.clone()
        return float_to_af_tensor(x_cloned, bits, stochastic=False)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_cloned = grad_output.clone()
        grad_input = float_to_af_tensor(
            grad_output_cloned,
            ctx.bits,
            stochastic=True,
            vector_chunk_size=ctx.chunk_size
        )
        return grad_input, None, None

class AetherLinearQAT(nn.Linear):
    def __init__(self, original_linear, bits, chunk_size):
        super().__init__(original_linear.in_features, original_linear.out_features, bias=original_linear.bias is not None)
        self.bits = bits
        self.chunk_size = chunk_size
        self.weight.data = original_linear.weight.data.clone()
        if original_linear.bias is not None:
            self.bias.data = original_linear.bias.data.clone()

    def forward(self, x):
        # Apply the Straight-Through Estimator with Shared SR on both W and X
        w_q = AetherSim.apply(self.weight, self.bits, self.chunk_size)
        x_q = AetherSim.apply(x, self.bits, self.chunk_size)
        return nn.functional.linear(x_q, w_q, self.bias)

def patch_model_qat(model, bits, chunk_size):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name != "lm_head":
            setattr(model, name, AetherLinearQAT(module, bits, chunk_size))
        else:
            patch_model_qat(module, bits, chunk_size)
    return model

def run_short_finetuning(label, dataset_tokens, all_start_indices, bits=16, chunk_size=16, steps=300):
    print(f"Starting QAT: {label} | Chunk Size: {chunk_size}")

    # Scale up to 7B with Multi-GPU Support
    print("Loading Qwen2.5-7B across available GPUs...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()

    if label != "BF16 Baseline":
        model = patch_model_qat(model, bits, chunk_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    loss_history = []
    model.train()

    seq_len = 256
    batch_size = 4

    for i in range(steps):
        start_indices = all_start_indices[i * batch_size : (i + 1) * batch_size]

        # Pin inputs strictly to the device of the first embedding layer
        input_device = model.model.embed_tokens.weight.device
        batch_inputs = torch.stack([dataset_tokens[idx : idx + seq_len] for idx in start_indices]).to(input_device)
        # HF AutoModelForCausalLM shifts labels internally (labels[:,1:] vs logits[:,:-1]).
        # Passing pre-shifted labels would cause a double-shift (t+2 prediction). Use clone().
        batch_labels = batch_inputs.clone()

        optimizer.zero_grad()
        outputs = model(batch_inputs, labels=batch_labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_history.append(loss.item())
        if i % 50 == 0:
            print(f"  Step {i:03d} | Loss: {loss.item():.4f}")

    del model
    torch.cuda.empty_cache()

    smoothed = [sum(loss_history[max(0, i-10):i+1])/len(loss_history[max(0, i-10):i+1]) for i in range(len(loss_history))]
    return smoothed

if __name__ == "__main__":
    torch.manual_seed(42)

    print("Loading tokenizer and WikiText dataset...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    text = "\n".join([t for t in dataset["text"][:5000] if len(t.strip()) > 10])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze()

    # Pre-generate shared batch indices so all four runs train on identical data,
    # decoupling data-loading RNG from the stochastic-rounding RNG in backward.
    _seq_len = 256
    _batch_size = 4
    _steps = 300
    _max_start = len(tokens) - _seq_len - 1
    all_start_indices = torch.randint(0, _max_start, (_steps * _batch_size,))

    torch.manual_seed(42)
    hist_bf16  = run_short_finetuning("BF16 Baseline", tokens, all_start_indices)
    torch.manual_seed(42)
    hist_ideal = run_short_finetuning("AF16", tokens, all_start_indices, chunk_size=1)
    torch.manual_seed(42)
    hist_ours  = run_short_finetuning("AF16", tokens, all_start_indices, chunk_size=16)
    torch.manual_seed(42)
    hist_break = run_short_finetuning("AF16", tokens, all_start_indices, chunk_size=256)

    plt.figure(figsize=(10, 6))
    plt.plot(hist_bf16, label='bfloat16 Baseline', linestyle='--', color='gray', linewidth=2)
    plt.plot(hist_ideal, label='AF16 (Chunk=1, Ideal SR)', alpha=0.5, color='blue')
    plt.plot(hist_ours, label='AF16 (Chunk=16, Proposed Hardware)', linewidth=2, color='green')
    plt.plot(hist_break, label='AF16 (Chunk=256, Over-correlated)', linestyle=':', color='red')

    plt.xlabel('Training Steps')
    plt.ylabel('Cross-Entropy Loss (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sr_ablation_qwen_real_7b.pdf', format='pdf', bbox_inches='tight')
    print("\nSaved sr_ablation_qwen_real_7b.pdf!")
