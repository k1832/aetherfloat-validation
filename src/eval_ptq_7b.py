import torch
import argparse
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
import warnings

warnings.filterwarnings("ignore")

from aether_sim import patch_model_ptq

def run_production_evaluation(fmt="bf16"):
    print(f"\n{'='*60}\nEvaluating Production LLM Scale | Format: {fmt.upper()}\n{'='*60}")

    model_id = "Qwen/Qwen2.5-7B"

    print(f"Loading {model_id} into VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Applying {fmt.upper()} Hardware Simulation PTQ...")
    if fmt == 'af16':
        model = patch_model_ptq(model, fmt='af', bits=16)
    elif fmt == 'af8':
        model = patch_model_ptq(model, fmt='af', bits=8)
    elif fmt == 'fp8':
        model = patch_model_ptq(model, fmt='fp8')

    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)

    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=["wikitext", "piqa", "hellaswag"]
    )

    print(f"\n--- Final Scale Results for {fmt.upper()} ---")
    print(f"  WikiText-2 PPL : {results['results']['wikitext']['word_perplexity,none']:.4f}")
    print(f"  PIQA Acc       : {results['results']['piqa']['acc,none']:.4f}")
    print(f"  HellaSwag Acc  : {results['results']['hellaswag']['acc,none']:.4f}\n")

    del lm_eval_model
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmt", type=str, default="all", choices=["bf16", "fp8", "af8", "af16", "all"])
    args = parser.parse_args()

    with torch.inference_mode():
        if args.fmt == "all":
            for f in ["bf16", "fp8", "af8", "af16"]:
                run_production_evaluation(f)
        else:
            run_production_evaluation(args.fmt)
