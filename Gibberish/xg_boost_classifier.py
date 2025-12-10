import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_curve,
    auc
)
from xgboost import XGBClassifier

def main():
    print("=== LOADING TRAIN & TEST DATA ===")

    train_df = pd.read_csv("train_mixed_1k.csv")
    test_df  = pd.read_csv("test_mixed_1k.csv")

    # --------------------------
    # Features
    # --------------------------
    features = [
        "LLC-load-misses",
        "dTLB-load-misses",
        "cycle_activity.stalls_total",
        "instructions",
        "cycles"
    ]

    # Labels
    y_train = train_df["True_Label"]
    y_test  = test_df["True_Label"]

    # Matrices
    X_train = train_df[features]
    X_test  = test_df[features]

    # --------------------------
    # CORRELATION HEATMAP
    # --------------------------
    plt.figure(figsize=(7, 6))
    corr = train_df[features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap for Gibbrish_1000")
    plt.tight_layout()
    plt.savefig("correlation_heatmap_1k.png")
    print("Saved: correlation_heatmap_1k.png")
    print("\n=== CORRELATION MATRIX ===")
    print(corr.to_string(), "\n")


    # --------------------------
    # SCALING
    # --------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("=== TRAINING XGBOOST CLASSIFIER ===")

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss"
    )

    clf.fit(X_train_scaled, y_train)

    # --------------------------
    # Predictions
    # --------------------------
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]  # probability for ROC

    print("=== RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    # --------------------------
    # CONFUSION MATRIX
    # --------------------------
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non‑Member", "Member"],
        yticklabels=["Non‑Member", "Member"]
    )
    plt.title("XGBoost Confusion Matrix for Gibbrish_1000")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("xgboost_confusion_matrix_1k.png")
    print("Saved: xgboost_confusion_matrix_1k.png")

    # --------------------------
    # ROC CURVE + AUC
    # --------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC‑AUC: {roc_auc:.3f}")


    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange",
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Gibbrish_1000")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve_1k.png")
    print("Saved: roc_curve_1k.png")

    # --------------------------
    # FEATURE IMPORTANCE
    # --------------------------
    importance = clf.feature_importances_
    print("Feature Importances:")
    for f, val in zip(features, importance):
        print(f"  {f}: {val:.4f}")

    plt.figure(figsize=(8, 6))
    plt.barh(features, importance, color="purple")
    plt.xlabel("Importance Score")
    plt.title("XGBoost Feature Importance for Gibbrish_1000")
    plt.tight_layout()
    plt.savefig("xgboost_feature_importance_1k.png")
    print("Saved: xgboost_feature_importance_1k.png")


if __name__ == "__main__":
    main()
