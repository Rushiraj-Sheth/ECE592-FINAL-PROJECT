import json
import random
import string
from tqdm import tqdm

# --- CONFIGURATION ---
# Output file for pure noise
OUTPUT_FILE = "pure_gibberish_noise.jsonl" 
NUM_SAMPLES = 10000 
MIN_LENGTH = 10
MAX_LENGTH = 50

def generate_noise(min_len, max_len):
    """
    Generates a string of random characters, digits, and punctuation.
    """
    # Includes uppercase, lowercase, digits, and symbols/punctuation
    characters = string.ascii_letters + string.digits + string.punctuation 
    
    length = random.randint(min_len, max_len)
    
    # Use random.choices for fast generation
    return ''.join(random.choices(characters, k=length))

def main():
    print(f"--- Generating {NUM_SAMPLES} samples of pure textual noise ---")
    
    data = []
    for _ in tqdm(range(NUM_SAMPLES)):
        gibberish_text = generate_noise(MIN_LENGTH, MAX_LENGTH)
        
        # Save in the same JSONL format as before ({"text": "..."})
        entry = {"text": gibberish_text}
        data.append(entry)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

    print(f"\nSUCCESS! Created {OUTPUT_FILE}")

if __name__ == "__main__":
    main()