import json

TRAIN_FILE = "pure_gibberish_noise.jsonl"
MEMBERS_FILE = "members_1000.txt"

def main():
    print("--- Loading Training Data ---")
    training_set = set()
    with open(TRAIN_FILE, 'r') as f:
        for line in f:
            # We load the JSON to get the "real" string, then strip whitespace
            data = json.loads(line)
            training_set.add(data['text'].strip())
            
    print(f"Loaded {len(training_set)} unique training samples.")

    print("\n--- Checking Members File ---")
    match_count = 0
    fail_count = 0
    
    with open(MEMBERS_FILE, 'r') as f:
        # Read lines and strip newline characters
        member_lines = [line.strip() for line in f if line.strip()]

    for i, member in enumerate(member_lines):
        if member in training_set:
            match_count += 1
        else:
            fail_count += 1
            if fail_count == 1:
                print(f"\n[MISMATCH EXAMPLE]")
                print(f"Member File: {repr(member)}")
                print("Reason: This string is NOT in the parsed JSON set.")

    print("-" * 30)
    print(f"Total Matches: {match_count}")
    print(f"Total Failures: {fail_count}")
    
    if fail_count == 0:
        print("\nSUCCESS: All members in members.txt exist in the training data.")
    else:
        print("\nFAILURE: Some members do not match. See example above.")

if __name__ == "__main__":
    main()