import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    # --- 1. Load the "Attack" Data ---
    print("Loading data from attack_non_member_results.csv and attack_member_results.csv...")
    
    try:
        # Load the "Non-Member" data (Vulnerable model + Non-Member prompt)
        non_member_df = pd.read_csv("non_member_results.csv")
        non_member_df['label'] = 0 # Label as '0'
        
        # Load the "Member" data (Vulnerable model + Member prompt)
        member_df = pd.read_csv("member_results.csv")
        member_df['label'] = 1 # Label as '1'
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure 'attack_non_member_results.csv' and 'member_results.csv' are in the same directory.")
        return

    # Combine them into one big dataset
    df = pd.concat([non_member_df, member_df], ignore_index=True)
    
    # Clean up any "ERROR" rows
    df.replace("ERROR", np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"\nTotal samples loaded: {len(df)} ({len(non_member_df)} non-member, {len(member_df)} member)")

    # --- 2. Statistical Analysis ---
    print("\n" + "="*30)
    print("--- Statistical Analysis (MLaaS Attack Scenario) ---")
    print("="*30)
    
    # Normalize our counters by 'instructions'
    df['LLC_misses_per_inst'] = df['LLC-load-misses'] / df['instructions']
    df['dTLB_misses_per_inst'] = df['dTLB-load-misses'] / df['instructions']
    df['stalls_per_inst'] = df['cycle_activity.stalls_total'] / df['instructions']
    df['ipc'] = df['instructions'] / df['cycles'] # Instructions Per Cycle (Efficiency)

    # Calculate the mean for each group
    analysis_cols = ['LLC_misses_per_inst', 'dTLB_misses_per_inst', 'stalls_per_inst', 'ipc']
    print(df.groupby('label')[analysis_cols].mean().to_markdown(floatfmt=".6f"))

    # --- 3. Visualization ---
    print("\nGenerating plots...")
    
    plot_df = df.melt(id_vars=['label'], value_vars=analysis_cols, var_name='Metric', value_name='Value')
    
    # Map labels to readable names
    plot_df['Membership'] = plot_df['label'].map({0: 'Non-Member Data', 1: 'Member Data'})
    
    plt.figure(figsize=(15, 8))
    # Note: We are plotting data from the SAME model, just with different data inputs
    sns.boxplot(data=plot_df, x='Metric', y='Value', hue='Membership', showfliers=False) 
    plt.title('MLaaS Attack: Hardware Footprint Comparison (Vulnerable Model)', fontsize=16)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xticks(rotation=10)
    
    plot_filename = 'attack_comparison_plot.png' # <-- New plot file
    plt.savefig(plot_filename)
    print(f"Saved plot to {plot_filename}")

    # --- 4. Build the "Attacker" ---
    print("\n" + "="*30)
    print("--- Attacker Model Training (MLaaS Scenario) ---")
    print("="*30)
    
    X = df[analysis_cols]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    attacker_model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Training attacker model on 80% of the data...")
    attacker_model.fit(X_train_scaled, y_train)

    # --- 5. Attacker Results ---
    print("\n" + "="*30)
    print("--- Attacker Model Results (MLaaS Scenario) ---")
    print("="*30)
    
    y_pred = attacker_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"MLaaS Attack Accuracy: {accuracy * 100:.2f}%")
    print("\nThis percentage represents the attacker's success rate in guessing")
    print("if a given prompt was 'member' or 'non-member' data.")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Non-Member (0)', 'Member (1)']))
    
    print("--- Confusion Matrix ---")
    print("(Rows = Actual, Columns = Predicted)")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()