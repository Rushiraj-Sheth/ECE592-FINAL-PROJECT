# ECE592-Project
MIA Project: The guide to run exp as below:

# SETUP DETAILS:
========================================================================
# 1. Create a virtual environment (do this once)
    $ python3 -m venv .venv
# 2. Activate it (do this every time you open this project)
    $ source .venv/bin/activate
# 3. Install the required libraries (do this once)
    $ pip install torch transformers datasets
========================================================================
# A. DATASET SETUP:
1. source .venv/bin/activate (IF NOT ACTIVATED & STARTING A NEW SESSION AGAIN)
2. $ python prepare_data.py
========================================================================
# B. FINE TUNING - small model & RUN THE EXP:
1. $ python fine_tune_cpu.py
2. $ pip install tqdm
3. $ pip install pandas scikit-learn matplotlib seaborn

4. $ perf stat -e LLC-load-misses:u,dTLB-load-misses:u,cycle_activity.stalls_total:u,instructions:u,cycles:u \
    python infer_cpu.py \
    --model_path "gpt2" \
    --prompt "Instruction: Write a Python function to add two numbers."   

5. $ perf stat -e LLC-load-misses:u,dTLB-load-misses:u,cycle_activity.stalls_total:u,instructions:u,cycles:u \
    python infer_cpu.py \
    --model_path "./gpt2-finetuned-python" \
    --prompt "Instruction: Write a Python function to add two numbers."     
**NOTE: THIS (4&5) ARE A CHECK FOR A SINGLE PROMPT; IF IT RUNS OR NOT. THIS ALSO CHECKS THE perf COUNTERS.
 IF YES, GO TO STEP-6


6. $ python collect_data.py \
    --model_path "./gpt2-finetuned-python" \
    --prompt "Instruction: Write a Python function to add two numbers." \
    --samples 100 \
    --output_csv "member_results.csv"

7. $ python collect_data.py \
    --model_path "./gpt2-finetuned-python" \
    --prompt "The capital of France is" \
    --samples 100 \
    --output_csv "non_member_results.csv"

8. $ python analyze_data.py
========================================================================
# C. FINE TUNE - medium model & RUN THE EXP:
1. $ python fine_tune_cpu_medium.py

2. $ python infer_cpu.py \
    --model_path "gpt2-medium" \
    --prompt "Instruction: Write a Python function to add two numbers."    

3. $ python infer_cpu.py \
    --model_path "./gpt2-medium-finetuned-python" \
    --prompt "Instruction: Write a Python function to add two numbers."        
"THIS IS A CHECK FOR A SINGLE PROMPT RUNS OR NOT. IF YES, GO TO STEP-4"

4. $ python collect_data.py \
    --model_path "./gpt2-medium-finetuned-python" \
    --prompt "Instruction: Write a Python function to add two numbers." \
    --samples 100 \
    --output_csv "member_medium_results.csv"

5. $ python collect_data.py \
    --model_path "./gpt2-medium-finetuned-python" \
    --prompt "The capital of France is" \
    --samples 100 \
    --output_csv "non_member_medium_results.csv"    

6. $ python analyze_data_medium.py
==========================================================================