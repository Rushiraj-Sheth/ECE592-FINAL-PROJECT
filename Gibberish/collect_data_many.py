import subprocess
import re
import csv
import argparse
from tqdm import tqdm
import os


PERF_COMMAND_BASE = [
    'perf', 'stat', '-e',
    'LLC-load-misses:u,dTLB-load-misses:u,cycle_activity.stalls_total:u,instructions:u,cycles:u'
]

INFER_SCRIPT = 'infer_cpu_clean.py'

COUNTER_NAMES = [
    "LLC-load-misses",
    "dTLB-load-misses",
    "cycle_activity.stalls_total",
    "instructions",
    "cycles"
]

PERF_PATTERN = re.compile(r"\s*([\d,]+)\s+([\w\.-]+:?u?)")

def run_single_measurement(model_path, prompt):
    command_to_run = PERF_COMMAND_BASE + [
        'python', INFER_SCRIPT,
        '--model_path', model_path,
        '--prompt', prompt
    ]

    result = subprocess.run(command_to_run, capture_output=True, text=True)
    stderr_output = result.stderr

    data = {}
    for line in stderr_output.splitlines():
        match = PERF_PATTERN.search(line)
        if match:
            value_str = match.group(1).replace(',', '')
            event_name = match.group(2).replace(':u', '')

            if event_name in COUNTER_NAMES:
                data[event_name] = int(value_str)

    return [data.get(name, "ERROR") for name in COUNTER_NAMES]

def main():
    parser = argparse.ArgumentParser(
        description="Run perf stat on MANY prompts from a file and save one row per prompt."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True, help="File with 1 prompt per line")
    parser.add_argument("--output_csv", type=str, required=True)

    args = parser.parse_args()

    print(f"Loading prompts from: {args.input_file}")
    prompts = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if clean:
                prompts.append(clean)

    print(f"Found {len(prompts)} prompts. Running profiling...")

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["prompt"] + COUNTER_NAMES)

        for prompt in tqdm(prompts, desc="Profiling"):
            try:
                row = run_single_measurement(args.model_path, prompt)
                writer.writerow([prompt] + row)
            except Exception as e:
                print(f"Error: {e}")
                writer.writerow([prompt] + ["ERROR"] * len(COUNTER_NAMES))

    print(f"Done! Saved to {args.output_csv}")

if __name__ == "__main__":
    main()
