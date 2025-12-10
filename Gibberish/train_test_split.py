import pandas as pd
from sklearn.model_selection import train_test_split

# Load your two datasets
members_df = pd.read_csv("results_members_1000.csv")
non_members_df = pd.read_csv("results_non_members_1000.csv")
# members_df = pd.read_csv("books3_member_perf.csv")
# non_members_df = pd.read_csv("books3_nonmember_perf.csv")

# Label them
members_df["True_Label"] = 1
non_members_df["True_Label"] = 0

# ---- Split Members ----
members_train, members_test = train_test_split(
    members_df, 
    test_size=500,   # keep 500 for testing
    random_state=42
)

# ---- Split Non-Members ----
non_members_train, non_members_test = train_test_split(
    non_members_df,
    test_size=500,   # keep 500 for testing
    random_state=42
)

# ---- Build Training Set (500 + 500 = 1000) ----
train_df = pd.concat([members_train, non_members_train]).sample(frac=1, random_state=42)

# ---- Build Testing Set (500 + 500 = 1000) ----
test_df = pd.concat([members_test, non_members_test]).sample(frac=1, random_state=42)

# Save them
train_df.to_csv("train_mixed_1k.csv", index=False)
test_df.to_csv("test_mixed_1k.csv", index=False)

print("DONE:")
print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")
