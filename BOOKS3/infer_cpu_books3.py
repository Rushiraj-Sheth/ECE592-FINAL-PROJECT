import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Deterministic CPU forward pass for perf.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load prompt from file
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    model.eval()
    model.to("cpu")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    )

    with torch.no_grad():
        _ = model(**inputs)

if __name__ == "__main__":
    main()
