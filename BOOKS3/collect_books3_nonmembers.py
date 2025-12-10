import hashlib
from datasets import load_dataset
from tqdm import tqdm

BOOKS3_DATASET = "Geralt-Targaryen/books3"
MEMBER_FILE = "books3_members_train.txt"
OUTPUT_FILE = "books3_nonmembers.txt"
NUM_SAMPLES = 5000

def normalize_text(t):
    """Clean and normalize whitespace so matching is consistent."""
    return t.replace("\n", " ").strip()

def hash_text(t):
    """Hash to make matching fast even for large texts."""
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

def main():
    print("Loading member text hashes...")
    member_hashes = set()

    with open(MEMBER_FILE, "r", encoding="utf-8") as f:
        for line in f:
            clean = normalize_text(line)
            member_hashes.add(hash_text(clean))

    print(f"Loaded {len(member_hashes)} member samples to skip.")

    print("Streaming Books3...")
    dataset = load_dataset(BOOKS3_DATASET, split="train", streaming=True)
    iterator = iter(dataset)

    nonmembers = []

    for _ in tqdm(range(NUM_SAMPLES * 3), desc="Searching for non-members"):
        # We oversample by ~3× to safely find enough unique non-members
        try:
            row = next(iterator)
        except StopIteration:
            break

        text = row["text"]
        if not isinstance(text, str) or len(text.strip()) < 20:
            continue

        clean = normalize_text(text)
        h = hash_text(clean)

        if h in member_hashes:
            # Skip member samples
            continue

        # Add to nonmember list
        nonmembers.append(clean)

        if len(nonmembers) >= NUM_SAMPLES:
            break

    print(f"Collected {len(nonmembers)} non-member samples.")
    print(f"Writing to {OUTPUT_FILE} ...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in nonmembers:
            f.write(line + "\n")

    print(f"Saved B_nonmember to: {OUTPUT_FILE}")
    print("\nNEXT STEP: Use these in your perf profiling (collect_data.py).")

if __name__ == "__main__":
    main()
