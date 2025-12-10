# Phase A — Books3 Membership Inference Pipeline

This document describes the exact sequence of scripts required to run the Books3
membership‑inference experiment.

---

## 1. Required Files

Environment & Setup:
Activate the Python virtual environment and ensure dependencies are installed.

```bash
# Activate Virtual Environment
source .venv/bin/activate

# Install Dependencies
pip install torch transformers datasets pandas scikit-learn matplotlib seaborn tabulate
```

Place the following scripts in the project root:

- `fine_tune_books3.py`
- `collect_books3_members.py`
- `save_books3_members.py`
- `collect_books3_nonmembers.py`
- `infer_cpu_clean.py`
- `collect_performance_books3.py`
- `rf_books3_membership.py`

Ensure Books3 dataset files are available in the expected paths used by the scripts.

---

## 2. Step‑by‑Step Execution

### **Step 1 — Fine‑tune the Model on Books3**

```bash
python3 fine_tune_books3.py
```

---

### **Step 2 — Save Books3 members used in finetuning**
```bash
python3 save_books3_members.py
```
This will genreate `books3_members_train.txt`


---

### **STEP 3 — Build the Books3 Non‑Member Set (B_nonmember)**
```bash
python3 collect_books3_nonmembers.py
```
This will genreate `books3_nonmembers.txt`

---

### **STEP 4 — Prepare Perf‑Ready Prompt Files for Profiling**
Now, we can collect perf counters readings for member and non-member data using `collect_performance_books3.py` script

FOR COLLECTING MEMBER PERF DATA
```bash
python collect_performance_books3.py \
    --model_path gpt2-medium-finetuned-books3 \
    --input_file books3_members_train.txt \
    --output_csv books3_member_perf.csv \
    --max_prompts 400

```

FOR COLLECTING NON-MEMBER PERF DATA
```bash
python collect_performance_books3.py \
    --model_path gpt2-medium-finetuned-books3 \
    --input_file books3_nonmembers.txt \
    --output_csv books3_nonmember_perf.csv \
    --max_prompts 400

```
These steps will generate the `books3_member_perf.csv` and `books3_nonmember_perf.csv` files.


---


### **STEP 5 — Build the Random Forest Classifier (Books3 Membership Inference)**

```bash
python rf_books3_membership.py
```
These will generate all the results specified in Phase A in the report.
