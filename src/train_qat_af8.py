import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
from aether_sim import patch_model_qat

def run_qat_finetuning(label, fmt, dataset_tokens, presampled_indices, bits=8, chunk_size=16, steps=200):
    print(f"Starting QAT: {label}")

    # 1. Scale up to 7B with Multi-GPU Support
    print("Loading Qwen2.5-7B across available GPUs...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto" # Shards the 7B model across all available GPUs to prevent OOM
    )

    # Enable Gradient Checkpointing to save massive amounts of VRAM during QAT
    model.gradient_checkpointing_enable()

    if fmt != "bf16":
        model = patch_model_qat(model, fmt, bits, chunk_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_history = []
    model.train()

    seq_len = 128
    batch_size = 4
    max_start = len(dataset_tokens) - seq_len - 1

    for i in range(steps):
        start_indices = presampled_indices[i]

        # Pin inputs strictly to the device of the first embedding layer.
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

    steps = 200
    seq_len = 128
    batch_size = 4
    max_start = len(tokens) - seq_len - 1

    # Pre-generate all batch indices once so every format trains on identical data.
    all_indices = torch.randint(0, max_start, (steps, batch_size))

    torch.manual_seed(42)
    hist_bf16 = run_qat_finetuning("BF16 Baseline", "bf16", tokens, all_indices)
    torch.manual_seed(42)
    hist_fp8  = run_qat_finetuning("FP8 E4M3 (Block Scaled)", "fp8", tokens, all_indices)
    torch.manual_seed(42)
    hist_af8  = run_qat_finetuning("AF8 (Scale-Free QAT)", "af", tokens, all_indices, bits=8, chunk_size=16)

    plt.figure(figsize=(10, 6))
    plt.plot(hist_bf16, label='bfloat16 Baseline', linestyle='--', color='gray', linewidth=2)
    plt.plot(hist_fp8, label='FP8 E4M3 + AMAX', color='orange', alpha=0.8, linewidth=2)
    plt.plot(hist_af8, label='AF8 (Block-Scale-Free Inference QAT)', color='red', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Cross-Entropy Loss (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('qat_8bit_convergence_ste_7b.pdf', format='pdf', bbox_inches='tight')
    print("\nSaved qat_8bit_convergence_ste_7b.pdf!")
