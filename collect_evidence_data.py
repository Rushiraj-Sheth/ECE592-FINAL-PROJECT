import subprocess
import re
import csv
import argparse
import os
from tqdm import tqdm

# --- CONFIGURATION (Must match your original experiment) ---
# The counters that successfully detected the 10k model
PERF_COMMAND_BASE = [
    'perf', 'stat', '-e', 
    'LLC-load-misses:u,dTLB-load-misses:u,cycle_activity.stalls_total:u,instructions:u,cycles:u',
]

INFER_SCRIPT = 'infer_cpu.py'

COUNTER_NAMES = [
    "LLC-load-misses",
    "dTLB-load-misses",
    "cycle_activity.stalls_total",
    "instructions",
    "cycles"
]

PERF_PATTERN = re.compile(r"\s*([\d,]+)\s+([\w\.-]+:?u?)")

def run_single_measurement(model_path, prompt):
    """
    Runs one 'perf stat' command for a specific prompt.
    """
    command_to_run = PERF_COMMAND_BASE + [
        'python', INFER_SCRIPT,
        '--model_path', model_path,
        '--prompt', prompt
    ]
    
    # Run the command and capture stderr (where perf writes)
    try:
        result = subprocess.run(command_to_run, capture_output=True, text=True)
        stderr_output = result.stderr
        
        # Parse counters
        data = {}
        for line in stderr_output.splitlines():
            match = PERF_PATTERN.search(line)
            if match:
                value_str = match.group(1).replace(',', '')
                event_name = match.group(2).replace(':u', '')
                if event_name in COUNTER_NAMES:
                    data[event_name] = int(value_str)
        
        return [data.get(name, "ERROR") for name in COUNTER_NAMES]
        
    except Exception as e:
        print(f"Subprocess Error: {e}")
        return ["ERROR"] * len(COUNTER_NAMES)

def main():
    parser = argparse.ArgumentParser(description="Run perf stat on a list of prompts from a file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--prompts_file", type=str, required=True, help="Text file with one prompt per line")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the CSV file")
    args = parser.parse_args()

    print(f"--- STARTING EVIDENCE COLLECTION ---")
    print(f"Model: {args.model_path}")
    print(f"Evidence Source: {args.prompts_file}")

    # 1. Read the Suspect Prompts
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(prompts)} suspect snippets to profile.")

    # 2. Run Profiling Loop
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(COUNTER_NAMES) # Header

        for prompt in tqdm(prompts, desc="Profiling Suspects"):
            # Run measurement
            row = run_single_measurement(args.model_path, prompt)
            writer.writerow(row)

    print(f"Done! Evidence saved to {args.output_csv}")

if __name__ == "__main__":
    main()