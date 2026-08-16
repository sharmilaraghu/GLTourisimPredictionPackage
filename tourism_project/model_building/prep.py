import os
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = "tourism_project/data/tourism.csv"

REPO_ROOT = "."

# 1. Load the dataset
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# 2. Clean unnecessary columns
# 'CustomerID' is just a unique identifier (leaks no predictive signal).
# 'Unnamed: 0' is the index column that pandas wrote when the CSV was saved.
cols_to_drop = ["CustomerID", "Unnamed: 0"]
existing_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=existing_to_drop)
print(f"Dropped columns: {existing_to_drop}")
print(f"Remaining columns: {list(df.columns)}")

# 3. Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# 4. Train/test split (stratified to preserve the class imbalance ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train target distribution:\n{y_train.value_counts()}")
print(f"Test target distribution:\n{y_test.value_counts()}")

# 5. Save the splits to the repo root so the workflow can upload them as artifacts.
X_train.to_csv(os.path.join(REPO_ROOT, "Xtrain.csv"), index=False)
X_test.to_csv(os.path.join(REPO_ROOT, "Xtest.csv"), index=False)
y_train.to_csv(os.path.join(REPO_ROOT, "ytrain.csv"), index=False)
y_test.to_csv(os.path.join(REPO_ROOT, "ytest.csv"), index=False)

print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv at repo root.")
print("Data preparation complete.")
