import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def add_stat_annotation(ax, data, x_col, y_col, order):
    """Helper to add median values on the plot"""
    medians = data.groupby(x_col)[y_col].median()
    for i, label in enumerate(order):
        val = medians[label]
        ax.text(i, val, f'{val:.4f}', 
                horizontalalignment='center', 
                verticalalignment='bottom', 
                fontweight='bold', color='black', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, pad=1))

def main():
    print("--- STEP 1: LOADING EVIDENCE (FINE-TUNED MODEL) ---")
    
    try:
        # Known Member (Python)
        control_df = pd.read_csv("attack_member_medium_10k_results.csv")
        control_df['Type'] = 'Control (Python)'
        control_df['True_Label'] = 1

        # Known Innocent (Wikitext)
        innocent_df = pd.read_csv("attack_non_member_medium_10k_results.csv")
        innocent_df['Type'] = 'Innocent (Wikitext)'
        innocent_df['True_Label'] = 0

        # Suspect Data (Books3)
        suspect_df = pd.read_csv("evidence_books3_suspect_results.csv")
        suspect_df['Type'] = 'Suspect (Books3)'
        suspect_df['True_Label'] = 0 
        
    except FileNotFoundError as e:
        print(f"ERROR: Missing file. {e}")
        return

    all_data = pd.concat([control_df, innocent_df, suspect_df]).reset_index(drop=True)

    print("\n--- STEP 2: PROCESSING METRICS ---")
    all_data['LLC Misses'] = all_data['LLC-load-misses'] / all_data['instructions']
    all_data['dTLB Misses'] = all_data['dTLB-load-misses'] / all_data['instructions']
    all_data['Stalls'] = all_data['cycle_activity.stalls_total'] / all_data['instructions']
    all_data['IPC'] = all_data['instructions'] / all_data['cycles']

    features = ['LLC Misses', 'dTLB Misses', 'Stalls', 'IPC']

    print("\n--- STEP 3: RUNNING RANDOM FOREST CLASSIFIER ---")
    
    # Prepare Training Data: 50% Control + 100% Innocent
    control_train, control_test = train_test_split(control_df, test_size=0.5, random_state=42)
    train_subset = pd.concat([control_train, innocent_df])
    
    # Re-calc features for training set
    for df in [train_subset]:
        df['LLC Misses'] = df['LLC-load-misses'] / df['instructions']
        df['dTLB Misses'] = df['dTLB-load-misses'] / df['instructions']
        df['Stalls'] = df['cycle_activity.stalls_total'] / df['instructions']
        df['IPC'] = df['instructions'] / df['cycles']

    X_train = train_subset[features]
    y_train = train_subset['True_Label']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    print("Classifier Trained.")
    
    # Predict on Suspect Data
    suspect_subset = all_data[all_data['Type'] == 'Suspect (Books3)']
    X_suspect = scaler.transform(suspect_subset[features])
    suspect_preds = clf.predict(X_suspect)
    
    # Predict on Control Test Set
    control_test['LLC Misses'] = control_test['LLC-load-misses'] / control_test['instructions']
    control_test['dTLB Misses'] = control_test['dTLB-load-misses'] / control_test['instructions']
    control_test['Stalls'] = control_test['cycle_activity.stalls_total'] / control_test['instructions']
    control_test['IPC'] = control_test['instructions'] / control_test['cycles']
    
    X_control = scaler.transform(control_test[features])
    control_preds = clf.predict(X_control)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT (FINE-TUNED MODEL)")
    print("="*50)
    print(f"[CONTROL CHECK] Python Code Detected: {(control_preds.sum()/len(control_preds))*100:.1f}%")
    print(f"[SUSPECT CHECK] Books3 Detected:      {(suspect_preds.sum()/len(suspect_preds))*100:.1f}%")
    print("="*50)

    print("\n--- STEP 4: GENERATING PLOTS ---")
    
    palette = {'Control (Python)': '#d62728', 'Innocent (Wikitext)': '#2ca02c', 'Suspect (Books3)': '#9467bd'}
    order = ['Control (Python)', 'Innocent (Wikitext)', 'Suspect (Books3)']

    # === PLOT 1: BOX PLOTS ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GPT_finetuned Hardware Profile: Box Plots', fontsize=20)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.boxplot(data=all_data, x='Type', y=feature, hue='Type', legend=False, palette=palette, ax=axes[i], showfliers=False, order=order)
        add_stat_annotation(axes[i], all_data, 'Type', feature, order) # Add Numbers
        axes[i].set_title(feature, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Normalized Value')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("GPT_finetuned_boxplots.png")
    print("Saved: GPT_finetuned_boxplots.png")

    # === PLOT 2: VIOLIN PLOTS ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GPT_finetuned Hardware Profile: Violin Plots', fontsize=20)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.violinplot(data=all_data, x='Type', y=feature, hue='Type', legend=False, palette=palette, ax=axes[i], inner="box", order=order)
        add_stat_annotation(axes[i], all_data, 'Type', feature, order) # Add Numbers
        axes[i].set_title(feature, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Normalized Value')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("GPT_finetuned_violinplots.png")
    print("Saved: GPT_finetuned_violinplots.png")

if __name__ == "__main__":
    main()