from datasets import load_dataset
import random

def main():
    print("Loading 'wikitext' (non-member) dataset...")
    # This is our general "non-member" data
    non_member_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')
    
    print("Loading 'iamtarun/python_code_instructions_18k_alpaca' (member) dataset...")
    # This is our specific "member" data (Python code)
    member_dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split='train')

    print("\n" + "="*30)
    print("--- Non-Member (Wikitext) Sample ---")
    print("="*30)
    # Filter out empty lines for a better sample
    wikitext_sample = [text for text in non_member_dataset['text'] if text.strip()][20] # Get a different line
    print(wikitext_sample)

    print("\n" + "="*30)
    print("--- Member (Python Code) Sample ---")
    print("="*30)
    # Get a random sample from the dataset
    random_index = random.randint(0, len(member_dataset) - 1)
    # This dataset has 'instruction' and 'output' columns. Let's show both.
    print("INSTRUCTION:")
    print(member_dataset[random_index]['instruction'])
    print("\nOUTPUT:")
    print(member_dataset[random_index]['output'])
    print("="*30)


if __name__ == "__main__":
    main()