# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import ttest_ind
# import os

# # --- CONFIGURATION ---
# FILE_MEMBERS = "results_members_1000.csv"
# FILE_NON_MEMBERS = "results_non_members_1000.csv"

# # The list of metrics to analyze
# METRICS_TO_PLOT = [
#     "LLC-load-misses",
#     "dTLB-load-misses",
#     "instructions",
#     "cycles",
#     "cycle_activity.stalls_total"
# ]
# # ---------------------

# def load_data(filepath, label):
#     """Loads CSV and adds a label column."""
#     if not os.path.exists(filepath):
#         print(f"Error: File {filepath} not found.")
#         return None
#     df = pd.read_csv(filepath)
#     df['Type'] = label
#     return df

# def analyze_single_metric(metric, df_mem, df_non):
#     print(f"Plotting {metric}...")

#     # 1. Clean Data
#     d_m = df_mem[df_mem[metric] != 'ERROR'].copy()
#     d_n = df_non[df_non[metric] != 'ERROR'].copy()

#     # Convert to numeric
#     d_m[metric] = pd.to_numeric(d_m[metric])
#     d_n[metric] = pd.to_numeric(d_n[metric])

#     # 2. Calculate Statistics
#     mean_mem = d_m[metric].mean()
#     mean_non = d_n[metric].mean()
    
#     # Percent Difference
#     diff_pct = ((mean_mem - mean_non) / mean_non) * 100
    
#     # T-Test (P-Value)
#     t_stat, p_val = ttest_ind(d_m[metric], d_n[metric], equal_var=False)

#     # Determine significance label for the graph
#     sig_label = "SIGNIFICANT" if p_val < 0.05 else "Insignificant"
    
#     # 3. Generate Plot
#     plt.figure(figsize=(10, 7))
    
#     # Combine data
#     combined = pd.concat([d_m, d_n])
    
#     # Plot Histogram
#     sns.histplot(
#         data=combined, 
#         x=metric, 
#         hue="Type", 
#         kde=True, 
#         element="step", 
#         bins=30, 
#         alpha=0.5
#     )
    
#     # --- ADD STATS TO TITLE ---
#     # We add the stats right into the title so it's impossible to miss
#     plt.title(
#         f"Metric: {metric}\n"
#         f"Difference: {diff_pct:+.2f}% | P-Value: {p_val:.5f} ({sig_label})",
#         fontsize=14, 
#         fontweight='bold'
#     )
    
#     plt.xlabel(f"{metric} (Lower is usually 'better/easier' for CPU)")
#     plt.ylabel("Frequency")
    
#     # Add a grid for readability
#     plt.grid(axis='y', linestyle='--', alpha=0.5)

#     # Save File
#     filename = f"plot_{metric}.png"
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.close()
#     print(f" -> Saved {filename}")

# def main():
#     # Load raw data
#     df_mem = load_data(FILE_MEMBERS, "Member")
#     df_non = load_data(FILE_NON_MEMBERS, "Non-Member")

#     if df_mem is None or df_non is None:
#         return

#     # Loop through metrics
#     for metric in METRICS_TO_PLOT:
#         try:
#             analyze_single_metric(metric, df_mem, df_non)
#         except KeyError:
#             print(f"Skipping {metric}: Column not found.")
#         except Exception as e:
#             print(f"Error on {metric}: {e}")

# if __name__ == "__main__":
#     main()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import os

# --- CONFIGURATION ---
FILE_MEMBERS = "results_members_1000.csv"
FILE_NON_MEMBERS = "results_non_members_1000.csv"

# <<< ADD THIS >>>
DATASET_NAME = "Gibberish_G1k"   # Change to "Gibberish_100", "Gibberish_1k", etc.

# HPC Metrics
METRICS_TO_PLOT = [
    "LLC-load-misses",
    "dTLB-load-misses",
    "instructions",
    "cycles",
    "cycle_activity.stalls_total"
]

def load_data(filepath, label):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None
    df = pd.read_csv(filepath)
    df['Type'] = label
    return df

# def analyze_single_metric(metric, df_mem, df_non):
#     print(f"Plotting {metric}...")

#     d_m = df_mem[df_mem[metric] != 'ERROR'].copy()
#     d_n = df_non[df_non[metric] != 'ERROR'].copy()

#     d_m[metric] = pd.to_numeric(d_m[metric])
#     d_n[metric] = pd.to_numeric(d_n[metric])

#     mean_mem = d_m[metric].mean()
#     mean_non = d_n[metric].mean()

#     diff_pct = ((mean_mem - mean_non) / mean_non) * 100

#     t_stat, p_val = ttest_ind(d_m[metric], d_n[metric], equal_var=False)

#     sig_label = "SIGNIFICANT" if p_val < 0.05 else "Insignificant"

#     plt.figure(figsize=(10, 7))
#     combined = pd.concat([d_m, d_n])

#     sns.histplot(
#         data=combined,
#         x=metric,
#         hue="Type",
#         kde=True,
#         element="step",
#         bins=30,
#         alpha=0.5
#     )

#     # <<< UPDATED TITLE WITH DATASET NAME >>>
#     plt.title(
#         f"{metric} for {DATASET_NAME} Dataset\n"
#         f"Diff: {diff_pct:+.2f}% | P-Value: {p_val:.5f} ({sig_label})",
#         fontsize=14,
#         fontweight='bold'
#     )

#     plt.xlabel(metric)
#     plt.ylabel("Frequency")
#     plt.grid(axis='y', linestyle='--', alpha=0.5)

#     filename = f"{DATASET_NAME}_plot_{metric}.png"
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.close()
#     print(f" -> Saved {filename}")

def analyze_single_metric(metric, df_mem, df_non):
    print(f"Plotting {metric}...")

    # --- Robust ERROR filtering ---
    d_m = df_mem[~df_mem[metric].astype(str).str.contains("ERR", case=False, na=True)].copy()
    d_n = df_non[~df_non[metric].astype(str).str.contains("ERR", case=False, na=True)].copy()

    d_m[metric] = pd.to_numeric(d_m[metric], errors="coerce")
    d_n[metric] = pd.to_numeric(d_n[metric], errors="coerce")

    d_m = d_m.dropna(subset=[metric])
    d_n = d_n.dropna(subset=[metric])

    if d_m.empty or d_n.empty:
        print(f" -> Skipping {metric}: No valid numeric data.")
        return

    # --- Compute stats ---
    mean_mem = d_m[metric].mean()
    mean_non = d_n[metric].mean()
    diff_pct = ((mean_mem - mean_non) / mean_non) * 100

    t_stat, p_val = ttest_ind(d_m[metric], d_n[metric], equal_var=False)
    sig_label = "SIGNIFICANT" if p_val < 0.05 else "Insignificant"

    plt.figure(figsize=(10, 7))
    combined = pd.concat([d_m, d_n])

    sns.histplot(
        data=combined,
        x=metric,
        hue="Type",
        kde=True,
        element="step",
        bins=50,
        alpha=0.5
    )

    # # --- Find KDE peaks for mem & non-mem ---
    # from scipy.signal import find_peaks

    # # get KDE curves
    # mem_kde = sns.kdeplot(d_m[metric], color="red", alpha=0).get_lines()[-1]
    # non_kde = sns.kdeplot(d_n[metric], color="blue", alpha=0).get_lines()[-1]

    # mem_x, mem_y = mem_kde.get_data()
    # non_x, non_y = non_kde.get_data()

    # peak_mem_idx = mem_y.argmax()
    # peak_non_idx = non_y.argmax()

    # peak_mem = mem_x[peak_mem_idx]
    # peak_non = non_x[peak_non_idx]

    # # --- Draw vertical lines ---
    # plt.axvline(peak_mem, color="red", linestyle="--", linewidth=1.6)
    # plt.axvline(peak_non, color="blue", linestyle="--", linewidth=1.6)

    # # --- Label the peaks ---
    # plt.text(
    #     peak_mem, plt.ylim()[0],
    #     f"{peak_mem:.1f}",
    #     color="red",
    #     ha="center", va="bottom",
    #     fontsize=10, fontweight="bold"
    # )
    # plt.text(
    #     peak_non, plt.ylim()[0],
    #     f"{peak_non:.1f}",
    #     color="blue",
    #     ha="center", va="bottom",
    #     fontsize=10, fontweight="bold"
    # )

    plt.title(
        f"{metric} for {DATASET_NAME} Dataset\n"
        f"Diff: {diff_pct:+.2f}% | P-Value: {p_val:.5f} ({sig_label})",
        fontsize=14,
        fontweight='bold'
    )

    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    filename = f"{DATASET_NAME}_plot_{metric}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f" -> Saved {filename}")



def main():
    df_mem = load_data(FILE_MEMBERS, "Member")
    df_non = load_data(FILE_NON_MEMBERS, "Non-Member")

    if df_mem is None or df_non is None:
        return

    for metric in METRICS_TO_PLOT:
        try:
            analyze_single_metric(metric, df_mem, df_non)
        except KeyError:
            print(f"Skipping {metric}: Column not found.")
        except Exception as e:
            print(f"Error on {metric}: {e}")

if __name__ == "__main__":
    main()

