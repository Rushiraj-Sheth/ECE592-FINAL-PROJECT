from datasets import load_dataset
from tqdm import tqdm

BOOKS3_DATASET = "Geralt-Targaryen/books3"
NUM_TRAIN_SAMPLES = 5000
OUTPUT_FILE = "books3_members_train.txt"

def main():
    print(f"Rebuilding B_train ({NUM_TRAIN_SAMPLES} samples) from Books3 stream...")

    dataset = load_dataset(BOOKS3_DATASET, split="train", streaming=True)
    iterator = iter(dataset)

    samples = []
    for _ in tqdm(range(NUM_TRAIN_SAMPLES), desc="Collecting member samples"):
        try:
            row = next(iterator)
            text = row["text"]

            if not isinstance(text, str) or len(text.strip()) < 20:
                continue

            # Save raw text (before tokenization)
            samples.append(text.strip())

        except StopIteration:
            break

    print(f"Collected {len(samples)} member samples.")
    print(f"Writing to {OUTPUT_FILE} ...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in samples:
            clean_line = line.replace("\n", " ").strip()
            f.write(clean_line + "\n")

    print(f"Saved B_train to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
