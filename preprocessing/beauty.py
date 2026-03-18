from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm
import json
import os

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = "beauty"
SEP = "::"
DATA_DIR = BASE_DIR / "data" / DATASET
FILE = DATA_DIR / "All_Beauty.jsonl.gz"
OUT_DIR = f"processed_{DATASET}"

# Generate Structure
os.makedirs(BASE_DIR / "data" / OUT_DIR, exist_ok=True)
os.makedirs(BASE_DIR / "data" / OUT_DIR / "nmf", exist_ok=True)
os.makedirs(BASE_DIR / "data" / OUT_DIR/ "bert", exist_ok=True)

# Load and match ml-1m style
df = pd.read_json(FILE, lines=True)
df = df.rename(columns={
    "user_id": "UserID",
    "asin": "ItemID",
    "rating": "Rating",
    "timestamp": "Timestamp"
})[["UserID", "ItemID", "Rating", "Timestamp"]]

df["UserID"] = df["UserID"].astype("category").cat.codes
df["ItemID"] = df["ItemID"].astype("category").cat.codes

#----------------------NeuMF----------------------------
# LOO
sorted_ratings = df.sort_values(by=["UserID", "Timestamp"], ascending=True)

item_rank = sorted_ratings.groupby("UserID").cumcount(ascending=False)
test_df = sorted_ratings[item_rank == 0].copy()
valid_df = sorted_ratings[item_rank == 1].copy()
train_df = sorted_ratings[item_rank >= 2].copy()

# Save data
nmf_train = train_df[["UserID", "ItemID", "Rating"]].to_numpy()
nmf_valid = valid_df[["UserID", "ItemID", "Rating"]].to_numpy()
nmf_test  = test_df[["UserID", "ItemID", "Rating"]].to_numpy()

with open(BASE_DIR / "data" / OUT_DIR / "nmf" / "train.pkl", "wb") as f:
    pickle.dump(nmf_train, f)
with open(BASE_DIR / "data" / OUT_DIR / "nmf" / "valid.pkl", "wb") as f:
    pickle.dump(nmf_valid, f)
with open(BASE_DIR / "data" / OUT_DIR / "nmf" / "test.pkl", "wb") as f:
    pickle.dump(nmf_test, f)

#------------------Bert4Rec-------------
user_history = sorted_ratings.groupby("UserID")["ItemID"].apply(list).to_dict()

bert_train_sequences = {}
bert_valid_sequences = {}
bert_test_sequences = {}

train_user_ids = []
valid_user_ids = []
test_user_ids = []


for user_id, items in tqdm(user_history.items(), desc="Splitting", total=len(user_history)):
    # Need at least 3 interactions
    if len(items) < 3:
        continue
    
    uid = int(user_id)

    # Train
    bert_train_sequences[uid] = {"seq": items[:-2]}
    train_user_ids.append(uid)
    
    # Valid
    bert_valid_sequences[uid] = {"seq": items[:-2], "target": items[-2]}
    valid_user_ids.append(uid)
    
    # Test
    bert_test_sequences[uid] = {"seq": items[:-1], "target": items[-1]}
    test_user_ids.append(uid)

# Save data
with open(BASE_DIR / "data" / OUT_DIR / "bert" / "train.pkl", "wb") as f:
    pickle.dump((bert_train_sequences, train_user_ids), f)
with open(BASE_DIR / "data" / OUT_DIR / "bert" / "valid.pkl", "wb") as f:
    pickle.dump((bert_valid_sequences, valid_user_ids), f)
with open(BASE_DIR / "data" / OUT_DIR / "bert" / "test.pkl", "wb") as f:
    pickle.dump((bert_test_sequences, test_user_ids), f)
