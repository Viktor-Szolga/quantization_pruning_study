from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm
import json
import os
import ast
import numpy as np
import gzip
import shutil
import random
np.random.seed(42)
random.seed(42)

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = "steam"
SEP = "::"
DATA_DIR = BASE_DIR / "data" / DATASET
ZIPPEDFILE = DATA_DIR / "steam_reviews.json.gz"
OUT_DIR = f"processed_{DATASET}"
FILE = ZIPPEDFILE.with_suffix("")
# Unzip file
if not os.path.isfile(FILE):   
    with gzip.open(ZIPPEDFILE, 'rb') as f_in:
        with open(FILE, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# Generate Structure
os.makedirs(BASE_DIR / "data" / OUT_DIR, exist_ok=True)
os.makedirs(BASE_DIR / "data" / OUT_DIR / "nmf", exist_ok=True)
os.makedirs(BASE_DIR / "data" / OUT_DIR/ "bert", exist_ok=True)

# Load and match ml-1m style
def parse_steam_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading", total=7_793_069):
            d = ast.literal_eval(line)
            hours = d.get('hours', 0)
            if hours is None: hours = 0
            
            if hours >= 4:
                yield {
                    'user_id': d.get('username'),
                    'item_id': d.get('product_id'),
                    'rating': 1.0,
                    'timestamp': d.get('date'),
                }
                
# Load into DataFrame
df = pd.DataFrame(parse_steam_data(FILE))
df = df.rename(columns={
    "user_id": "UserID",
    "item_id": "ItemID",
    "rating": "Rating",
    "timestamp": "Timestamp"
})[["UserID", "ItemID", "Rating", "Timestamp"]]

user_counts = df["UserID"].value_counts()
df = df[df["UserID"].isin(user_counts[user_counts >= 5].index)]
df = df.drop_duplicates(subset=["UserID", "ItemID"], keep="first")

df["UserID"] = df["UserID"].astype("category").cat.codes + 1
df["ItemID"] = df["ItemID"].astype("category").cat.codes + 1

# Stable item mapping like ml-1m
unique_items = sorted(df["ItemID"].unique())
print(min(unique_items))
item2id = {item: idx + 1 for idx, item in enumerate(unique_items)}
df["ItemID"] = df["ItemID"].map(item2id)
#----------------------NeuMF----------------------------
# LOO
sorted_ratings = df.sort_values(by=["UserID", "Timestamp"], ascending=True)

item_rank = sorted_ratings.groupby("UserID").cumcount(ascending=False)
test_df = sorted_ratings[item_rank == 0].copy()
valid_df = sorted_ratings[item_rank == 1].copy()
train_df = sorted_ratings[item_rank >= 2].copy()

print(df["UserID"].min())
print(df["ItemID"].min())
num_users = df["UserID"].max()
num_items = len(unique_items)

stats = (num_users, num_items)

item_counts = train_df["ItemID"].value_counts()

popularity = np.zeros(num_items + 1, dtype=np.float64)  # index 0 = padding
for item_id, count in item_counts.items():
    popularity[item_id] = count

popularity_smooth = popularity #** 0.75
popularity_smooth = popularity_smooth / popularity_smooth.sum()

user_history = sorted_ratings.groupby("UserID")["ItemID"].apply(list).to_dict()
user_item_set = {
    int(user_id): set(items)
    for user_id, items in user_history.items()
}

with open(BASE_DIR / "data" / OUT_DIR / "popularity.pkl", "wb") as f:
    pickle.dump({
        "counts": popularity,
        "prob": popularity_smooth,
        "user_item_set": user_item_set
    }, f)

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
with open(BASE_DIR / "data" / OUT_DIR / "stats.pkl", "wb") as f:
    pickle.dump(stats, f)
#------------------Bert4Rec-------------
user_history = sorted_ratings.groupby("UserID")["ItemID"].apply(list).to_dict()

bert_train_sequences = {}
bert_valid_sequences = {}
bert_test_sequences = {}

train_user_ids = []
valid_user_ids = []
test_user_ids = []


for user_id, items in tqdm(user_history.items(), desc="Splitting", total=len(user_history)):
    # Need at least 5 interactions
    if len(items) < 5:
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
with open(BASE_DIR / "data" / OUT_DIR / "stats.pkl", "wb") as f:
    pickle.dump(stats, f)