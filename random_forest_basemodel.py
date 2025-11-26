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
    print("--- STEP 1: LOADING EVIDENCE (BASE MODEL) ---")
    
    try:
        guilty_df = pd.read_csv("attack_member_medium_10k_results.csv")
        guilty_df['Type'] = 'Ref: FT-Python'
        guilty_df['True_Label'] = 1

        innocent_df = pd.read_csv("attack_non_member_medium_10k_results.csv")
        innocent_df['Type'] = 'Ref: FT-Wikitext'
        innocent_df['True_Label'] = 0

        # Test Subject: Base Model
        base_model_df = pd.read_csv("evidence_base_model_books3.csv")
        base_model_df['Type'] = 'TEST: Base-Books3'
        base_model_df['True_Label'] = 0 
        
    except FileNotFoundError as e:
        print(f"ERROR: Missing file. {e}")
        return

    all_data = pd.concat([guilty_df, innocent_df, base_model_df]).reset_index(drop=True)

    print("\n--- STEP 2: PROCESSING METRICS ---")
    all_data['LLC Misses'] = all_data['LLC-load-misses'] / all_data['instructions']
    all_data['dTLB Misses'] = all_data['dTLB-load-misses'] / all_data['instructions']
    all_data['Stalls'] = all_data['cycle_activity.stalls_total'] / all_data['instructions']
    all_data['IPC'] = all_data['instructions'] / all_data['cycles']

    features = ['LLC Misses', 'dTLB Misses', 'Stalls', 'IPC']

    print("\n--- STEP 3: RUNNING CLASSIFICATION TEST ---")
    
    # Train Judge
    ref_data = all_data[all_data['Type'].isin(['Ref: FT-Python', 'Ref: FT-Wikitext'])]
    test_data = all_data[all_data['Type'] == 'TEST: Base-Books3']

    X_train = ref_data[features]
    y_train = ref_data['True_Label']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Test Base Model
    X_test = scaler.transform(test_data[features])
    base_preds = clf.predict(X_test)
    
    detection_rate = (base_preds.sum() / len(base_preds)) * 100

    print("\n" + "="*50)
    print("BASE MODEL ANALYSIS REPORT")
    print("="*50)
    print(f"Base Model (Books3) Similarity to Guilty Profile: {detection_rate:.1f}%")
    print("="*50)

    print("\n--- STEP 4: GENERATING PLOTS ---")
    
    palette = {
        'Ref: FT-Python': '#d62728',     # Red
        'Ref: FT-Wikitext': '#2ca02c',   # Green
        'TEST: Base-Books3': '#1f77b4'   # Blue
    }
    order = ['Ref: FT-Python', 'Ref: FT-Wikitext', 'TEST: Base-Books3']

    # === PLOT 1: BOX PLOTS ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GPT_base Hardware Profile: Box Plots', fontsize=20)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.boxplot(data=all_data, x='Type', y=feature, hue='Type', legend=False, palette=palette, ax=axes[i], showfliers=False, order=order)
        add_stat_annotation(axes[i], all_data, 'Type', feature, order)
        axes[i].set_title(feature, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Normalized Value')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("GPT_base_boxplots.png")
    print("Saved: GPT_base_boxplots.png")

    # === PLOT 2: VIOLIN PLOTS ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GPT_base Hardware Profile: Violin Plots', fontsize=20)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.violinplot(data=all_data, x='Type', y=feature, hue='Type', legend=False, palette=palette, ax=axes[i], inner="box", order=order)
        add_stat_annotation(axes[i], all_data, 'Type', feature, order)
        axes[i].set_title(feature, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Normalized Value')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("GPT_base_violinplots.png")
    print("Saved: GPT_base_violinplots.png")

if __name__ == "__main__":
    main()