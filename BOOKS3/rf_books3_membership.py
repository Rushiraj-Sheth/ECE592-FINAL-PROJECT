import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----- CONFIG -----
MEMBER_CSV = "books3_member_perf.csv"
NONMEMBER_CSV = "books3_nonmember_perf.csv"
OUTPUT_PREFIX = "books3_mi_results"
EXP_TITLE = "Books3 Membership Inference Experiment"
# ------------------

def main():
    print("Loading perf data...")

    # Load raw perf data
    df_mem = pd.read_csv(MEMBER_CSV)
    df_non = pd.read_csv(NONMEMBER_CSV)

    df_mem["label"] = 1
    df_non["label"] = 0

    df = pd.concat([df_mem, df_non], ignore_index=True)

    # Remove invalid rows
    for col in df.columns:
        df = df[df[col] != "ERROR"]

    df = df.apply(pd.to_numeric)

    X = df.drop(columns=["label"])
    y = df["label"]

    print(f"Dataset size: {len(df)} samples")

    # ======================================================
    # TERMINAL STATISTICS — MEMBER VS NON-MEMBER
    # ======================================================
    member_means = df[df.label == 1].mean()
    nonmember_means = df[df.label == 0].mean()
    percent_change = ((member_means - nonmember_means) / nonmember_means) * 100

    print("\n=== PER-COUNTER STATISTICS (RAW VALUES) ===")
    for feat in X.columns:
        print(f"{feat}: Member mean={member_means[feat]:.2f}, "
              f"Non-member mean={nonmember_means[feat]:.2f}, "
              f"% change={percent_change[feat]:.2f}%")
    print("==========================================\n")

    # ======================================================
    # NORMALIZATION (unchanged)
    # ======================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ======================================================
    # ORIGINAL TRAIN/TEST SPLIT (accuracy preserved)
    # ======================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # --------------------------
    # TRAIN RF MODEL (unchanged)
    # --------------------------
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # --------------------------
    # METRICS
    # --------------------------
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    features = X.columns

    # ======================================================
    # 1. CORRELATION HEATMAP
    # ======================================================
    plt.figure(figsize=(7, 6))
    corr = X.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{EXP_TITLE} — Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_correlation_heatmap.png")
    plt.close()

    # ======================================================
    # 2. HISTOGRAMS WITH % CHANGE LABELS
    # ======================================================
    for feat in features:
        plt.figure(figsize=(7, 5))
        sns.histplot(df[df.label == 1][feat], color="red",
                     kde=True, stat="density", label="Member")
        sns.histplot(df[df.label == 0][feat], color="blue",
                     kde=True, stat="density", label="Non-Member")
        plt.legend()
        plt.xlabel(feat)

        pct = percent_change[feat]
        text = f"% change: {pct:.2f}%"
        plt.text(
            0.02, 0.95, text,
            transform=plt.gca().transAxes,
            fontsize=12,
            ha="left",
            va="top",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )

        plt.title(f"{EXP_TITLE} — Distribution of {feat}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_PREFIX}_hist_{feat}.png")
        plt.close()

    # ======================================================
    # 3. ROC CURVE
    # ======================================================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{EXP_TITLE} — ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_roc_curve.png")
    plt.close()

    # ======================================================
    # 4. CONFUSION MATRIX
    # ======================================================
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Member", "Member"],
        yticklabels=["Non-Member", "Member"]
    )
    plt.title(f"{EXP_TITLE} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_confusion_matrix.png")
    plt.close()

    # ======================================================
    # 5. FEATURE IMPORTANCE
    # ======================================================
    plt.figure(figsize=(8, 5))
    sns.barplot(x=clf.feature_importances_, y=features, color="purple")
    plt.xlabel("Importance Score")
    plt.title(f"{EXP_TITLE} — Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_feature_importance.png")
    plt.close()

    print("\nSaved ALL Books3 extended plots.\n")

if __name__ == "__main__":
    main()
