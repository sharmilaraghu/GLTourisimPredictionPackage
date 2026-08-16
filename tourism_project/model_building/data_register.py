import os
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

# Expected columns as they appear in the data dictionary in Project.md
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

# 1. Check that the file exists
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at: {DATA_PATH}")
    sys.exit(1)

# 2. Load the dataset
df = pd.read_csv(DATA_PATH)

# 3. Validate columns
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

if missing:
    print(f"[ERROR] Missing expected columns: {missing}")
    sys.exit(1)

if extra:
    print(f"[WARNING] Unexpected extra columns (will be ignored): {extra}")

# 4. Print a short summary
print("=" * 60)
print("DATASET REGISTRATION SUMMARY")
print("=" * 60)
print(f"File path            : {DATA_PATH}")
print(f"Rows                 : {df.shape[0]}")
print(f"Columns              : {df.shape[1]}")
print(f"Column validation    : OK (all {len(EXPECTED_COLUMNS)} expected columns present)")
print("-" * 60)
print("Target distribution (ProdTaken):")
print(df["ProdTaken"].value_counts().rename({0: "Not purchased", 1: "Purchased"}))
print("-" * 60)
print("Missing values per column:")
print(df.isnull().sum())
print("-" * 60)
print("Column dtypes:")
print(df.dtypes)
print("-" * 60)
print("First 5 rows:")
print(df.head())
print("=" * 60)
print("Dataset registered successfully.")
