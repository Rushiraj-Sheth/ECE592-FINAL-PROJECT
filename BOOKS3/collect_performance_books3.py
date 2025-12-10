import subprocess
import re
import csv
import argparse
from tqdm import tqdm
import tempfile
import os

# -----------------------------
# PERF EVENTS (same as before)
# -----------------------------
PERF_COMMAND_BASE = [
    'perf', 'stat', '-e',
    'LLC-load-misses:u,dTLB-load-misses:u,cycle_activity.stalls_total:u,instructions:u,cycles:u'
]

# Use the CLEAN deterministic inference script
INFER_SCRIPT = 'infer_cpu_books3.py'

COUNTER_NAMES = [
    "LLC-load-misses",
    "dTLB-load-misses",
    "cycle_activity.stalls_total",
    "instructions",
    "cycles"
]

# Parse perf output
PERF_PATTERN = re.compile(r"\s*([\d,]+)\s+([\w\.-]+:?u?)")

# ---------------------------------------------------------
# Run perf for ONE prompt by writing prompt → temp file
# ---------------------------------------------------------
def run_single_measurement(model_path, prompt):

    # Write prompt to a temp file to avoid argument-length limits
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(prompt)
        tmp_path = tmp.name

    command_to_run = PERF_COMMAND_BASE + [
        'python', INFER_SCRIPT,
        '--model_path', model_path,
        '--prompt_file', tmp_path
    ]

    result = subprocess.run(command_to_run, capture_output=True, text=True)
    stderr_output = result.stderr

    # Remove temp file
    os.unlink(tmp_path)

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


# ---------------------------------------------------------
# MAIN: loop through all prompts in the input file
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Perf-side-channel collector for Books3 MIA")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--max_prompts", type=int, default=500,
                        help="Limit number of prompts to profile (default=500)")
    args = parser.parse_args()

    print(f"Loading prompts from: {args.input_file}")

    # Load all prompts
    prompts = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if clean:
                prompts.append(clean)

    print(f"Found {len(prompts)} total prompts.")

    # Limit sample size for faster experiment
    if len(prompts) > args.max_prompts:
        prompts = prompts[:args.max_prompts]

    print(f"Using {len(prompts)} prompts for profiling.\n")

    # Open output CSV
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(COUNTER_NAMES)

        for prompt in tqdm(prompts, desc="Profiling"):
            try:
                row = run_single_measurement(args.model_path, prompt)
                writer.writerow(row)
            except Exception as e:
                print(f"Error: {e}")
                writer.writerow(["ERROR"] * len(COUNTER_NAMES))

    print(f"\nDone! Saved perf data to: {args.output_csv}")


if __name__ == "__main__":
    main()
