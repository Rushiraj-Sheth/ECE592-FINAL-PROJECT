import subprocess
import re
import csv
import argparse
import os
from tqdm import tqdm

# This is the 'perf' command we've been using, broken into a list
PERF_COMMAND_BASE = [
    'perf', 'stat', '-e', 
    'LLC-load-misses:u,dTLB-load-misses:u,cycle_activity.stalls_total:u,instructions:u,cycles:u',
]

# This is the python script we are targeting
INFER_SCRIPT = 'infer_cpu.py'

# This is the list of counters we want to find and save, in order.
# This MUST match the 'perf stat' command above.
COUNTER_NAMES = [
    "LLC-load-misses",
    "dTLB-load-misses",
    "cycle_activity.stalls_total",
    "instructions",
    "cycles"
]

# A regex pattern to find a number (with or without commas) and its event name
# e.g., "   123,456,789      LLC-load-misses"
PERF_PATTERN = re.compile(r"\s*([\d,]+)\s+([\w\.-]+:?u?)")

def run_single_measurement(model_path, prompt):
    """
    Runs one 'perf stat' command on our 'infer_cpu.py' script.
    Parses the stderr and returns the 5 counter values.
    """
    # Build the full command to run, e.g.:
    # ['perf', 'stat', ..., 'python', 'infer_cpu.py', '--model_path', 'gpt2', ...]
    command_to_run = PERF_COMMAND_BASE + [
        'python', INFER_SCRIPT,
        '--model_path', model_path,
        '--prompt', prompt
    ]
    
    # Run the command. Note: 'perf stat' writes its output to stderr.
    result = subprocess.run(command_to_run, capture_output=True, text=True)
    stderr_output = result.stderr

    # Parse the stderr output to find our numbers
    # We store them in a dictionary for easy lookup
    data = {}
    for line in stderr_output.splitlines():
        match = PERF_PATTERN.search(line)
        if match:
            # Value is Group 1 (e.g., "123,456,789")
            value_str = match.group(1).replace(',', '') # Remove commas
            
            # Name is Group 2 (e.g., "LLC-load-misses:u")
            event_name = match.group(2).replace(':u', '') # Clean the name
            
            if event_name in COUNTER_NAMES:
                data[event_name] = int(value_str)
                
    # Return the data in the correct, ordered list for the CSV
    return [data.get(name, "ERROR") for name in COUNTER_NAMES]

def main():
    parser = argparse.ArgumentParser(description="Run 'perf stat' multiple times and save to CSV.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--prompt", type=str, required=True, help="The input prompt")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to collect")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the CSV file")
    args = parser.parse_args()

    print(f"Starting data collection for model: {args.model_path}")
    print(f"Collecting {args.samples} samples. This will take a while...")

    # Open the CSV file and write the header row
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(COUNTER_NAMES) # Write the header

        # Run the loop N times, with a progress bar
        for _ in tqdm(range(args.samples), desc="Collecting samples"):
            try:
                # Run one measurement
                row = run_single_measurement(args.model_path, args.prompt)
                
                # Write the row of 5 numbers to our CSV
                writer.writerow(row)
            except Exception as e:
                print(f"Error during a run: {e}")
                writer.writerow(["ERROR"] * len(COUNTER_NAMES))

    print(f"Done! Data saved to {args.output_csv}")

if __name__ == "__main__":
    main()