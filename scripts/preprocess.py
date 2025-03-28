import pandas as pd
import hashlib
from pathlib import Path

# ---- CONFIG ----
INPUT_FILE = "data/creditcard.csv"
OUTPUT_DIR = "data/splits"

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ---- STEP 1: Load Dataset ----
print("📥 Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# ---- STEP 2: Normalize 'Time' and 'Amount' ----
print("🧮 Normalizing 'Time' and 'Amount'...")
df["Time_scaled"] = (df["Time"] - df["Time"].mean()) / df["Time"].std()
df["Amount_scaled"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

# ---- STEP 3: Create hash buckets from 'Time' column ----
def hash_to_bucket(value, num_buckets=10):
    """Hash a value and assign it to a deterministic bucket."""
    hash_val = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(hash_val, 16) % num_buckets

print("🔀 Applying deterministic hash split...")
df["bucket"] = df["Time"].apply(hash_to_bucket)

# ---- STEP 4: Assign train/val/test based on hash bucket ----
df_train = df[df["bucket"] < 7].drop(columns=["bucket"])
df_val   = df[df["bucket"] == 7].drop(columns=["bucket"])
df_test  = df[df["bucket"] > 7].drop(columns=["bucket"])

# ---- STEP 5: Save to CSV ----
print("💾 Saving train/val/test splits to CSV...")
df_train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
df_val.to_csv(f"{OUTPUT_DIR}/validation.csv", index=False)
df_test.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

print("✅ Done! Splits saved in 'data/splits/' folder.")
