import torch
from transformers import pipeline, AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run a single CPU inference to profile.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the model (local dir or Hugging Face ID)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True, 
        help="The input prompt for the model"
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_path} (on CPU)")
    
    # Initialize the text-generation pipeline
    # device=-1 is the explicit command for "USE CPU ONLY"
    generator = pipeline(
        'text-generation', 
        model=args.model_path, 
        tokenizer=args.model_path,
        trust_remote_code=True,
        device=-1  
    )
    
    # Set the pad token ID (our gpt2-finetuned model has it, base gpt2 might not)
    if generator.model.config.pad_token_id is None:
        generator.model.config.pad_token_id = generator.tokenizer.eos_token_id

    print(f"Running inference on prompt: \"{args.prompt[:50]}...\"")
    
    # --- THIS IS THE ACTION WE WILL PROFILE ---
    # Run a single, small text generation
    outputs = generator(
        args.prompt,
        max_new_tokens=50,  # Just generate 50 tokens
        num_return_sequences=1
    )
    # ------------------------------------------

    print("\n--- Model Output ---")
    print(outputs[0]['generated_text'])
    print("--------------------")

if __name__ == "__main__":
    main()