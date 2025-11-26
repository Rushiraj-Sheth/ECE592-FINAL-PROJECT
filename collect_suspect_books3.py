import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os

# --- CONFIGURATION ---
# The "Aggressive" Dataset: A mirror of the banned Books3
SUSPECT_DATASET = "Geralt-Targaryen/books3" 
MODEL_PATH = "./gpt2-medium-finetuned-python" # Point to your 10k model
OUTPUT_CSV = "evidence_books3_suspect_results.csv"
SAMPLES_TO_COLLECT = 100

def get_device():
    return "cpu"

def main():
    print(f"--- STARTING FORENSIC COLLECTION: {SUSPECT_DATASET} ---")
    print("WARNING: You are accessing a dataset of copyrighted/pirated books.")
    
    # 1. Load the Suspect Data (Streaming Mode to avoid 300GB download)
    print("Connecting to Shadow Library Mirror...")
    try:
        dataset = load_dataset(SUSPECT_DATASET, split="train", streaming=True)
    except Exception as e:
        print(f"FATAL ERROR: Could not access Books3 mirror. Reason: {e}")
        print("Backup Plan: Use 'monology/pile-uncopyrighted' (subset 'books3' if available) or 'wikitext'.")
        return

    # 2. Load Your "Witness" Model
    print(f"Loading Witness Model: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        model.to(get_device())
        model.eval()
    except OSError:
        print("Error: 10k Model not found. Please ensure training finished.")
        return

    # 3. The Interrogation Loop
    print(f"Extracting {SAMPLES_TO_COLLECT} snippets from suspect books...")
    
    results = []
    iterator = iter(dataset)
    
    for i in tqdm(range(SAMPLES_TO_COLLECT), desc="Scanning Books"):
        try:
            # Grab a book
            data = next(iterator)
            text = data['text']
            
            # Take a "fingerprint" (first 100 tokens)
            # We don't want the whole book, just enough to trigger recognition
            inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            prompt_str = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            
            # --- RUN HARDWARE PROFILING (Simulated Wrapper) ---
            # NOTE: In a real run, this loop happens inside 'collect_data.py' via perf.
            # Since we need to PIPE this text into 'collect_data.py', we will save the PROMPTS first.
            
            # Ideally, we save these PROMPTS to a file, then run collect_data.py on them.
            # But to be aggressive, let's just save the text so you can see what we found.
            results.append(prompt_str)
            
        except StopIteration:
            break

    # 4. Save the "Suspect List"
    # We will save these prompts to a text file. 
    # YOU MUST RUN 'collect_data.py' using these prompts to get hardware counters.
    
    suspect_file = "suspect_prompts_books3.txt"
    with open(suspect_file, "w", encoding="utf-8") as f:
        for prompt in results:
            # Clean newlines for command line usage
            clean_prompt = prompt.replace("\n", " ").replace('"', '').strip()[:200]
            f.write(clean_prompt + "\n")
            
    print(f"\nSUCCESS. Captured {len(results)} suspect book snippets.")
    print(f"Saved to: {suspect_file}")
    print("Next Step: Feed these prompts into your 'collect_data.py' script.")

if __name__ == "__main__":
    main()