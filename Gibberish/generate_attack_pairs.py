import json
import random
import string

# --- CONFIGURATION ---
SOURCE_TRAINING_FILE = "pure_gibberish_noise.jsonl" # Your existing training file
MEMBER_OUTPUT = "members_1000.txt"                       # Output file for members
NON_MEMBER_OUTPUT = "non_members_1000.txt"               # Output file for non-members
NUM_SAMPLES = 1000                                   # How many pairs to create
# ---------------------

def generate_gibberish_like(reference_text):
    """
    Creates a random string with the EXACT length and character set  and 
    of the reference text.
    """
    length = len(reference_text)
    
    # We use all printable characters to match your noisy dataset
    chars = string.ascii_letters + string.digits + string.punctuation + " "
    
    return "".join(random.choice(chars) for _ in range(length))

def main():
    print(f"Reading ALL training data from {SOURCE_TRAINING_FILE}...")
    
    # 1. Load the existing Members (Training Data)
    all_members = []
    try:
        with open(SOURCE_TRAINING_FILE, 'r') as f:
            for line in f:
                # Parse JSONL to get just the text string
                data = json.loads(line)
                all_members.append(data['text'])
    except FileNotFoundError:
        print(f"Error: Could not find {SOURCE_TRAINING_FILE}")
        return

    # 2. Pick random samples to test
    # Even though you trained on ALL of them, we only need to profile a subset (e.g., 100).
    selected_members = random.sample(all_members, min(NUM_SAMPLES, len(all_members)))
    
    print(f"Selected {len(selected_members)} members for profiling.")

    # 3. Create the Twin Non-Members
    # For every member, we create a fake one of the exact same length.
    generated_non_members = []
    
    for member_text in selected_members:
        fake_text = generate_gibberish_like(member_text)
        generated_non_members.append(fake_text)

    # 4. Save to plain text files (ready for batch_profiler.py)
    print("Writing output files...")
    
    with open(MEMBER_OUTPUT, 'w') as f_mem:
        f_mem.write("\n".join(selected_members))
        
    with open(NON_MEMBER_OUTPUT, 'w') as f_non:
        f_non.write("\n".join(generated_non_members))

    print("Success!")
    print(f"1. {MEMBER_OUTPUT} (Real training data)")
    print(f"2. {NON_MEMBER_OUTPUT} (Fake data, same length)")

if __name__ == "__main__":
    main()