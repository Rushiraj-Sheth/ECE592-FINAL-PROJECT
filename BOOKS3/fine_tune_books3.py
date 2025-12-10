import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from tqdm import tqdm
import os

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_BASE = "gpt2-medium"
MODEL_FINETUNE_DIR = "./gpt2-medium-finetuned-books3"

# Books3 mirror dataset (streaming)
BOOKS3_DATASET = "Geralt-Targaryen/books3"

# Number of samples to use for fine-tuning (member set)
NUM_TRAIN_SAMPLES = 5000
MAX_LENGTH = 128  # keep consistent with profiling
# -------------------------------------------------------

def stream_books3_samples():
    """Stream Books3 and extract NUM_TRAIN_SAMPLES short text snippets."""
    print(f"Streaming Books3 ({BOOKS3_DATASET})...")
    dataset = load_dataset(BOOKS3_DATASET, split="train", streaming=True)
    iterator = iter(dataset)

    samples = []
    for _ in tqdm(range(NUM_TRAIN_SAMPLES), desc="Collecting Books3 fine-tuning samples"):
        try:
            row = next(iterator)
            text = row["text"]

            if not isinstance(text, str) or len(text.strip()) < 20:
                continue

            samples.append({"text": text})
        except StopIteration:
            break

    print(f"Collected {len(samples)} Books3 samples.")
    return samples


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def main():
    print(f"\n--- Loading GPT-2 Medium base model ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    model = AutoModelForCausalLM.from_pretrained(MODEL_BASE)
    # model = model.to("cuda")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    # -------------------------------------------------------
    # STEP 1 — Build B_train from Books3
    # -------------------------------------------------------
    samples = stream_books3_samples()

    # HuggingFace expects datasets; make a simple one
    from datasets import Dataset
    dataset = Dataset.from_list(samples)

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    print(f"Prepared {len(tokenized)} tokenized Books3 training samples.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=MODEL_FINETUNE_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        no_cuda=False,
        disable_tqdm=False,
        # ADD THIS LINE:
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("\n--- Starting CPU Fine-Tuning on Books3 ---")
    trainer.train()
    print("\n--- Fine-Tuning Complete ---")

    print(f"Saving final model to {MODEL_FINETUNE_DIR}")
    model.save_pretrained(MODEL_FINETUNE_DIR)
    tokenizer.save_pretrained(MODEL_FINETUNE_DIR)

    print("\n>>> Books3 fine-tuned model ready.\n")


if __name__ == "__main__":
    main()
