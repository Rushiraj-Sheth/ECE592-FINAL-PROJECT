import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import os

# --- Configuration ---
MODEL_BASE = "gpt2-medium"  # <-- THE ONLY CHANGE: Use the "large" model
MODEL_FINETUNE_DIR = "./gpt2-medium-finetuned-python" # <-- New save directory
DATASET_NAME = "iamtarun/python_code_instructions_18k_alpaca"
# ---------------------

def formatting_prompts_func(example):
    text = f"Instruction: {example['instruction']}\nOutput: {example['output']}"
    return {"text": text}

def main():
    print(f"Loading base model: {MODEL_BASE}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        print("Tokenizer has no pad_token. Adding a new one.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"New pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    print(f"Loading and formatting dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(formatting_prompts_func)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    print("Tokenizing dataset...")
    # We will use the SAME 2,000 samples for a fair comparison
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    ).shuffle(seed=42).select(range(2000)) 

    print(f"Using a subset of {len(tokenized_dataset)} samples for CPU training.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=MODEL_FINETUNE_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=8,  
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        no_cuda=True,  # Force CPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print(f"--- Starting Fine-Tuning (on CPU) for {MODEL_BASE} ---")
    trainer.train()
    print("--- Fine-Tuning Complete ---")

    print(f"Saving fine-tuned model to {MODEL_FINETUNE_DIR}")
    model.save_pretrained(MODEL_FINETUNE_DIR)
    tokenizer.save_pretrained(MODEL_FINETUNE_DIR)
    print("CPU model fine-tuning complete.")

if __name__ == "__main__":
    main()