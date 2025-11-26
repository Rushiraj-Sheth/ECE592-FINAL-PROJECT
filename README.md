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
(Before tuning try running `pip install 'accelerate>=0.26.0' `  )
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

# Hardware Side-Channel Forensic Analysis: Detecting Copyrighted Data in LLMs

This project demonstrates a hardware side-channel attack (using `perf stat`) to determine if a Large Language Model (LLM) was trained on specific copyrighted data. We use a **10,000-sample fine-tuned GPT-2 Medium** model as our test subject and "Books3" (pirated books) as our suspect dataset.

## 📂 Required Scripts List
Ensure the following scripts are present in the root of your repository:
1. `fine_tune_cpu_medium.py` (Configured for 10k samples)
2. `collect_data.py` (Generic single-prompt collector)
3. `collect_suspect_books3.py` (Downloads suspect text snippets)
4. `collect_evidence.py` (Profiles a list of prompts from a file)
5. `forensic_finetuned_analysis.py` (Analyzes the Fine-Tuned Model behavior)
6. `validate_pretraining_plots.py` (Analyzes the Base Model Control behavior)

---

## 🚀 Phase 1: Environment & Setup

Activate the Python virtual environment and ensure dependencies are installed.

```bash
# Activate Virtual Environment
source .venv/bin/activate

# Install Dependencies
pip install torch transformers datasets pandas scikit-learn matplotlib seaborn tabulate
```
---

## 🏗 Phase 2: Fine-Tuning the "Witness" Model
We fine-tune gpt2-medium on 10,000 samples of Python code. This creates a strong "Member" signal in the hardware (cache/pipeline) that we can use as a reference.

### Step 1: Run the Training

```bash
python fine_tune_cpu_medium.py
```
Output: A new folder ./gpt2-medium-finetuned-python containing the model.
Note: This process takes several hours on a CPU.

---

## 📊 Phase 3: Establishing the Baseline (The Calibration)
We need to teach our classifier what "Guilty" (Member) and "Innocent" (Non-Member) hardware traces look like.

### Step 1: Collect "Guilty" Traces (Python Code) Using the fine-tuned model to read Python.

```bash
python collect_data.py \
    --model_path "./gpt2-medium-finetuned-python" \
    --prompt "Instruction: Write a Python function to add two numbers." \
    --samples 100 \
    --output_csv "attack_member_medium_10k_results.csv"
```

### Step 2: Collect "Innocent" Traces (Wikitext) Using the fine-tuned model to read Wikipedia (which it was NOT fine-tuned on).

```bash
python collect_data.py \
    --model_path "./gpt2-medium-finetuned-python" \
    --prompt "The capital of France is" \
    --samples 100 \
    --output_csv "attack_non_member_medium_10k_results.csv"
```

---

## Phase 4: The Forensic Investigation (Suspect Data)
Now we introduce the "Suspect" data: The Books3 dataset (pirated books). We want to see if the model recognizes this data.

### Step 1: Download Suspect Evidence This script downloads 100 snippets of pirated books and saves them to `suspect_prompts_books3.txt`.

```bash
python collect_suspect_books3.py
```

### Step 2: Profile Suspect Data on the Fine-Tuned Model We record the hardware traces while the Fine-Tuned Model reads the books.

```bash
python collect_evidence.py \
    --model_path "./gpt2-medium-finetuned-python" \
    --prompts_file "suspect_prompts_books3.txt" \
    --output_csv "evidence_books3_suspect_results.csv"
```

### Step 3: Step 3: Run the Forensic Analysis (The First Verdict) This script uses a Random Forest to classify the Books3 traces as "Guilty" or "Innocent."

```bash
python forensic_finetuned_analysis.py
```

Outputs:
A. `finetuned_forensic_boxplots.png`
B. `finetuned_forensic_violinplots.png`
C. Terminal Result: "Suspect Detection Rate" (e.g., 100%).

---

## Phase 5: The Scientific Validation (Control Test)
If Phase 4 shows a 100% detection rate, we must verify if this is due to our fine-tuning or if the Base Model (raw GPT-2) already knew the books.

### Step 1: Profile Suspect Data on the BASE Model We record the hardware traces while the raw `gpt2-medium` reads the books.

```bash
python collect_evidence.py \
    --model_path "gpt2-medium" \
    --prompts_file "suspect_prompts_books3.txt" \
    --output_csv "evidence_base_model_books3.csv"
```

### Step 2: Run the Validation Analysis This compares the Base Model's reaction against the known "Guilty" profile.

```bash
python validate_pretraining_plots.py
```

Outputs:
    `validation_boxplots.png`
    `validation_violinplots.png`
    Terminal Result: "Base Model Similarity to Guilty Profile."

---

## Conclusion Interpretation

### If Phase 4 is High (~100%) AND Phase 5 is Low (~0%):
Verdict: The fine-tuning process successfully leaked the membership data. The Base Model does not know the data.

### If Phase 4 is High (~100%) AND Phase 5 is High (>80%):
Verdict: Pre-Training Memory Detected. The model recognized the copyrighted books from its original pre-training phase. The fine-tuning did not overwrite this deep memory. (This is the stronger research finding).