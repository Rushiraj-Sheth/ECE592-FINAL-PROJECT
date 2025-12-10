# Phase B — Gibberish Membership Inference Pipeline

This document describes the exact sequence of scripts required to run the Gibberish
membership-inference experiment.

---

## 1. Required Files

Environment & Setup:
Activate the Python virtual environment and ensure dependencies are installed.

```bash
# Activate Virtual Environment
source .venv/bin/activate

# Install Dependencies
pip install -r requirements.txt

```

Place the following scripts in the project root:

- `pure_gibberish_data.py`
- `gpt_2_medium_tuning_gpu.py`
- `generate_attack_pairs.py`
- `collect_data_many.py`
- `infer_cpu_clean.py`
- `analyze_all_metrics_with_stats.py`
- `train_test_split.py`
- `xg_boost_classifier.py`

Ensure Gibberish dataset files are available in the expected paths used by the scripts.

---

## 2. Step-by-Step Execution


### **Step 0 — Generate Gibberish Dataset**

```bash
python3 pure_gibberish_data.py
```
This will generate the dataset of 10000 samples. which is present inside the file name "pure_gibberish_noise.jsonl"

### **Step 1 — Fine-tune the Model on Gibberish**

```bash
python3 gpt_2_medium_tuning_gpu.py
```
This will tune the Gibberish dataset for all 10000 data set for 20 epoches. We will be running on GPU for this one. Initially we were running on CPU which took time. Output folder will be generated "gpt2-medium-finetuned-pure-gibberish"
---

### **Step 2 — Generating Attack Pairs**
```bash
python3 save_books3_members.py
```
This will make `members_1000.txt` and `non_members_1000.txt` from `pure_gibberish_noise.jsonl`
But the Non member is not from pure_gibberish_noise.jsonl file.


STEP CAN VERIFY MEM AND NON MEM


### **STEP 3 — Prepare Perf-Ready Prompt Files for Profiling**
Now, we can collect perf counters readings for member and non-member data using `collect_performance_books3.py` script

FOR COLLECTING MEMBER PERF DATA
```bash
python collect_data_many.py   --model_path "./gpt2-medium-finetuned-pure-gibberish"   --input_file "members_1000.txt"   --output_csv "results_members_1000.csv"

```

FOR COLLECTING NON-MEMBER PERF DATA
```bash
python collect_data_many.py   --model_path "./gpt2-medium-finetuned-pure-gibberish"   --input_file "non_members_1000.txt"   --output_csv "results_non_members_1000.csv"
```
These steps will generate the `results_members_1000.csv` and `results_non_members_1000.csv` files.



### **Step 4 — Analysis of the perf counters via Histogram plots**
```bash
python3 analyze_all_metrics_with_stats.py
```
This will genreate All the 5 Figures of each perf counter for member and non-member.


python3 analyze_all_metrics_with_stats.py
Plotting LLC-load-misses...
 -> Saved Gibberish_G1k_plot_LLC-load-misses.png
Plotting dTLB-load-misses...
 -> Saved Gibberish_G1k_plot_dTLB-load-misses.png
Plotting instructions...
 -> Saved Gibberish_G1k_plot_instructions.png
Plotting cycles...
 -> Saved Gibberish_G1k_plot_cycles.png
Plotting cycle_activity.stalls_total...
 -> Saved Gibberish_G1k_plot_cycle_activity.stalls_total.png



### **Step 5 — Generating the training and testing set for Classifier .**
```bash
python3 train_test_split.py
```
This will generate the `train_mixed_1k.csv` and `test_mixed_1k.csv` from the `results_members_1000.csv` and `results_non_members_1000.csv`.



python3 train_test_split.py
DONE:
Training set size: 1000
Testing set size: 1000



### **Step 6 — Analysis of the XG Boost Classifier by training and testing.**
```bash
python3 xg_boost_classifier.py
```
This will generate the performance of the classifier by plotting ROC-AUC curve, Confusion matrix etc..



python3 xg_boost_classifier.py
=== LOADING TRAIN & TEST DATA ===
Saved: correlation_heatmap_1k.png

=== CORRELATION MATRIX ===
                             LLC-load-misses  dTLB-load-misses  cycle_activity.stalls_total  instructions    cycles
LLC-load-misses                     1.000000          0.449405                    -0.565581     -0.683287 -0.612371
dTLB-load-misses                    0.449405          1.000000                    -0.261007     -0.353189 -0.289211
cycle_activity.stalls_total        -0.565581         -0.261007                     1.000000      0.974320  0.997587
instructions                       -0.683287         -0.353189                     0.974320      1.000000  0.985789
cycles                             -0.612371         -0.289211                     0.997587      0.985789  1.000000

=== TRAINING XGBOOST CLASSIFIER ===
=== RESULTS ===
Accuracy: 0.730
              precision    recall  f1-score   support

           0       0.71      0.78      0.74       500
           1       0.76      0.68      0.71       500

    accuracy                           0.73      1000
   macro avg       0.73      0.73      0.73      1000
weighted avg       0.73      0.73      0.73      1000

Confusion Matrix:
 [[392 108]
 [162 338]]
Saved: xgboost_confusion_matrix_1k.png
ROC-AUC: 0.827
Saved: roc_curve_1k.png
Feature Importances:
  LLC-load-misses: 0.2785
  dTLB-load-misses: 0.1426
  cycle_activity.stalls_total: 0.1574
  instructions: 0.2678
  cycles: 0.1537
Saved: xgboost_feature_importance_1k.png

















