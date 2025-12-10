# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForLanguageModeling
# )
# from datasets import load_dataset
# import os

# # --- Configuration ---
# MODEL_BASE = "gpt2-medium"
# MODEL_FINETUNE_DIR = "./gpt2-medium-finetuned-pure-gibberish"
# LOCAL_DATA_FILE = "pure_gibberish_noise.jsonl" 
# # ---------------------

# def main():
#     # 1. Check for GPU availability
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Hardware check: Training will run on **{device.upper()}**")
    
#     if device == "cpu":
#         print("WARNING: No GPU detected. Training will be slow.")
#     else:
#         print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

#     print(f"Loading base model: {MODEL_BASE}")
    
#     # Load Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
    
#     # Load Model (Trainer will automatically move this to GPU during setup)
#     model = AutoModelForCausalLM.from_pretrained(MODEL_BASE, trust_remote_code=True)

#     # GPT-2 padding fix
#     if tokenizer.pad_token is None:
#         print("Adding pad_token...")
#         tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#         model.resize_token_embeddings(len(tokenizer))
#         model.config.pad_token_id = tokenizer.pad_token_id

#     print(f"Loading local dataset from: {LOCAL_DATA_FILE}")
    
#     # --- LOAD LOCAL FILE ---
#     try:
#         dataset = load_dataset("json", data_files=LOCAL_DATA_FILE, split="train")
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return

#     def tokenize_function(examples):
#         # Increased max_length slightly as GPUs handle it better, 
#         # but kept to 128 for speed if your VRAM is low.
#         return tokenizer(
#             examples["text"], 
#             truncation=True, 
#             padding="max_length", 
#             max_length=128 
#         )

#     print("Tokenizing dataset...")
#     tokenized_dataset = dataset.map(
#         tokenize_function, 
#         batched=True, 
#         remove_columns=dataset.column_names
#     )

#     train_dataset = tokenized_dataset.shuffle(seed=42)
#     print(f"Training on {len(train_dataset)} samples.")

#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#     print("Setting up training arguments (GPU Mode)...")
#     training_args = TrainingArguments(
#         output_dir=MODEL_FINETUNE_DIR,
#         num_train_epochs=1,
        
#         # GPU Optimizations:
#         per_device_train_batch_size=8,  # Increased from 2 (GPUs usually handle more)
#         gradient_accumulation_steps=1,   # Adjusted since batch size is larger
        
#         # Critical GPU Flags
#         fp16=True,                       # Uses Mixed Precision (much faster on NVIDIA GPUs)
#         # use_cpu=False,                 # Default is False, so we just remove the True flag
        
#         save_strategy="no",
#         logging_steps=10,
#         report_to="none",
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         data_collator=data_collator,
#     )

#     print(f"--- Starting Fine-Tuning ---")
#     trainer.train()
#     print("--- Fine-Tuning Complete ---")

#     print(f"Saving model to {MODEL_FINETUNE_DIR}")
#     model.save_pretrained(MODEL_FINETUNE_DIR)
#     tokenizer.save_pretrained(MODEL_FINETUNE_DIR)
#     print("Success!")

# if __name__ == "__main__":
#     main()



# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForLanguageModeling
# )
# from datasets import load_dataset
# import sys

# # --- Configuration ---
# MODEL_BASE = "gpt2-medium"
# MODEL_FINETUNE_DIR = "./gpt2-medium-finetuned-pure-gibberish"
# LOCAL_DATA_FILE = "pure_gibberish_noise.jsonl" 
# # ---------------------

# def main():
#     print(f"--- Hardware Check ---")
#     if torch.cuda.is_available():
#         print(f"GPU: {torch.cuda.get_device_name(0)}")
#     else:
#         print("CRITICAL: No GPU found.")
#         return

#     # 1. Load Tokenizer
#     print(f"Loading Tokenizer: {MODEL_BASE}")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
    
#     # CRITICAL FIX 1: Do NOT add a new token. Reuse EOS.
#     tokenizer.pad_token = tokenizer.eos_token
    
#     # 2. Load Model with "Eager" Attention
#     # CRITICAL FIX 2: 'attn_implementation="eager"' prevents SDPA/FlashAttn crashes on A100
#     print(f"Loading Model: {MODEL_BASE}")
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_BASE, 
#         trust_remote_code=True,
#         attn_implementation="eager" 
#     )
#     model.config.pad_token_id = model.config.eos_token_id

#     # 3. Load & Process Dataset
#     print(f"Loading dataset: {LOCAL_DATA_FILE}")
#     dataset = load_dataset("json", data_files=LOCAL_DATA_FILE, split="train")

#     def tokenize_function(examples):
#         return tokenizer(
#             examples["text"], 
#             truncation=True, 
#             padding="max_length", 
#             max_length=128
#         )

#     print("Tokenizing...")
#     tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
#     train_dataset = tokenized_dataset.shuffle(seed=42)

#     # CRITICAL FIX 3: Verify Data before GPU
#     print("--- Verifying Dataset Integrity ---")
#     vocab_size = model.config.vocab_size
#     print(f"Model Vocab Size: {vocab_size}")
    
#     # Check the first 100 samples for out-of-bounds tokens
#     for i in range(min(100, len(train_dataset))):
#         sample = train_dataset[i]['input_ids']
#         max_id = max(sample)
#         if max_id >= vocab_size:
#             print(f"FATAL ERROR: Found token ID {max_id} which is >= vocab size {vocab_size}")
#             print("This causes the CUDA Device-Side Assert.")
#             print("Solution: Clear your HuggingFace cache (rm -rf ~/.cache/huggingface/datasets)")
#             sys.exit(1)
#     print("Data verification passed. IDs are safe.")

#     # 4. Training Arguments
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#     print("Setting up training arguments...")
#     training_args = TrainingArguments(
#         output_dir=MODEL_FINETUNE_DIR,
#         num_train_epochs=1,
#         per_device_train_batch_size=64, 
#         gradient_accumulation_steps=1,
        
#         # A100 Settings
#         bf16=True,                       
#         fp16=False,
        
#         save_strategy="no",
#         logging_steps=10,
#         report_to="none",
#         dataloader_num_workers=4,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         data_collator=data_collator,
#     )

#     print(f"--- Starting Fine-Tuning ---")
#     trainer.train()
#     print("--- Fine-Tuning Complete ---")

#     model.save_pretrained(MODEL_FINETUNE_DIR)
#     tokenizer.save_pretrained(MODEL_FINETUNE_DIR)

# if __name__ == "__main__":
#     main()



# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForLanguageModeling
# )
# from datasets import load_dataset
# import os

# # --- Configuration ---
# MODEL_BASE = "gpt2-medium"
# MODEL_FINETUNE_DIR = "./gpt2-medium-finetuned-pure-gibberish"
# LOCAL_DATA_FILE = "pure_gibberish_noise.jsonl" 
# # ---------------------

# def main():
#     print(f"Loading base model: {MODEL_BASE} on CPU")
    
#     # Load Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
    
#     # Load Model (CPU default)
#     model = AutoModelForCausalLM.from_pretrained(MODEL_BASE, trust_remote_code=True)

#     # GPT-2 padding fix
#     if tokenizer.pad_token is None:
#         print("Adding pad_token...")
#         tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#         model.resize_token_embeddings(len(tokenizer))
#         model.config.pad_token_id = tokenizer.pad_token_id

#     print(f"Loading local dataset from: {LOCAL_DATA_FILE}")
    
#     # --- LOAD LOCAL FILE ---
#     try:
#         dataset = load_dataset("json", data_files=LOCAL_DATA_FILE, split="train")
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         print("Make sure 'synthetic_semantic_dataset.jsonl' is in this folder!")
#         return

#     print(f"Sample loaded: {dataset[0]}")

#     def tokenize_function(examples):
#         # SPEED OPTIMIZATION FOR CPU:
#         # Reduced max_length to 128 so it finishes quickly.
#         # If you want "real" training later, change to 512.
#         return tokenizer(
#             examples["text"], 
#             truncation=True, 
#             padding="max_length", 
#             max_length=128 
#         )

#     print("Tokenizing dataset...")
#     tokenized_dataset = dataset.map(
#         tokenize_function, 
#         batched=True, 
#         remove_columns=dataset.column_names
#     )

#     # For a quick test, let's just use 100 samples. 
#     # If this works, you can remove the .select() part to use all data.
#     train_dataset = tokenized_dataset.shuffle(seed=42)          #.select(range(min(100, len(tokenized_dataset))))
#     print(f"Training on {len(train_dataset)} samples for testing.")

#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#     print("Setting up training arguments (CPU Mode)...")
#     training_args = TrainingArguments(
#         output_dir=MODEL_FINETUNE_DIR,
#         num_train_epochs=1,
#         per_device_train_batch_size=2,   # Keep small for RAM safety
#         gradient_accumulation_steps=4,
#         save_strategy="no",
#         logging_steps=5,
#         report_to="none",
#         # --- CPU FORCED SETTINGS ---
#         use_cpu=False,      # Explicitly state CPU usage
#         use_cuda=True,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         data_collator=data_collator,
#     )

#     print(f"--- Starting Fine-Tuning ---")
#     trainer.train()
#     print("--- Fine-Tuning Complete ---")

#     print(f"Saving model to {MODEL_FINETUNE_DIR}")
#     model.save_pretrained(MODEL_FINETUNE_DIR)
#     tokenizer.save_pretrained(MODEL_FINETUNE_DIR)
#     print("Success!")

# if __name__ == "__main__":
#     main()


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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Configuration ---
MODEL_BASE = "gpt2-medium"
MODEL_FINETUNE_DIR = "./gpt2-medium-finetuned-pure-gibberish"
LOCAL_DATA_FILE = "pure_gibberish_noise.jsonl" 
# ---------------------

def main():
    # 1. Hardware Check
    if torch.cuda.is_available():
        print(f"Hardware: {torch.cuda.get_device_name(0)}")
    else:
        print("CRITICAL: No GPU found. The training will fail or be slow.")
    
    print(f"Loading base model: {MODEL_BASE}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
    
    # --- CRITICAL STABILITY FIX ---
    # DO NOT add a new [PAD] token. It causes CUDA crashes on resizing.
    # We reuse the End-Of-Sentence (EOS) token for padding.
    tokenizer.pad_token = tokenizer.eos_token
    # ------------------------------

    # Load Model
    # attn_implementation="eager" is safer for A100s on older models like GPT-2
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE, 
        trust_remote_code=True,
        attn_implementation="eager" 
    )
    model.config.pad_token_id = model.config.eos_token_id

    print(f"Loading local dataset from: {LOCAL_DATA_FILE}")
    
    try:
        dataset = load_dataset("json", data_files=LOCAL_DATA_FILE, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=128 
        )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )

    train_dataset = tokenized_dataset.shuffle(seed=42)
    print(f"Training on {len(train_dataset)} samples.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Setting up training arguments (GPU Mode)...")
    # training_args = TrainingArguments(
    #     output_dir=MODEL_FINETUNE_DIR,
    #     num_train_epochs=20,
        
    #     # --- A100 OPTIMIZATIONS ---
    #     per_device_train_batch_size=32,  # Good balance for A100
    #     gradient_accumulation_steps=1,
    #     fp16=True,                       # Enable Mixed Precision (Much faster)
    #     bf16=False,                      # Keep False for GPT-2 stability
        
    #     # --- GPU SETTINGS ---
    #     # use_cpu=False,                 # Default is False, no need to list it
    #     # use_cuda=True,                 # REMOVED (This caused your error)
        
    #     dataloader_num_workers=0,        # Prevent data loading freezes
    #     save_strategy="no",
    #     logging_steps=10,
    #     report_to="none",
    #     no_cuda=False,
    #     disable_tqdm=False,        
    # )

    training_args = TrainingArguments(
    output_dir=MODEL_FINETUNE_DIR,
    num_train_epochs=20,

    # ---- GPU Optimizations ----
    per_device_train_batch_size=8,   # 32 may explode VRAM depending on sequence length
    gradient_accumulation_steps=2,
    fp16=True,
    bf16=False,

    # ---- Learning Rate ----
    learning_rate=3e-5,              # GOOD starting point
    warmup_steps=200,                # Helps stabilize early steps

    # ---- Logging ----
    logging_steps=20,
    save_strategy="epoch",
    report_to="none",

    # ---- GPU Setting ----
    no_cuda=False,
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(f"--- Starting Fine-Tuning ---")
    trainer.train()
    print("--- Fine-Tuning Complete ---")

    print(f"Saving model to {MODEL_FINETUNE_DIR}")
    model.save_pretrained(MODEL_FINETUNE_DIR)
    tokenizer.save_pretrained(MODEL_FINETUNE_DIR)
    print("Success!")

if __name__ == "__main__":
    main()