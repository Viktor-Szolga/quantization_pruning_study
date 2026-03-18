from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm
import json
import os
import ast

import gzip
import shutil

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

df["UserID"] = df["UserID"].astype("category").cat.codes
df["ItemID"] = df["ItemID"].astype("category").cat.codes

#----------------------NeuMF----------------------------
# LOO
sorted_ratings = df.sort_values(by=["UserID", "Timestamp"], ascending=True)

item_rank = sorted_ratings.groupby("UserID").cumcount(ascending=False)
test_df = sorted_ratings[item_rank == 0].copy()
valid_df = sorted_ratings[item_rank == 1].copy()
train_df = sorted_ratings[item_rank >= 2].copy()


num_users = sorted_ratings["UserID"].max()
num_items = sorted_ratings["ItemID"].max()

stats = (num_users, num_items)

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
with open(BASE_DIR / "data" / OUT_DIR / "stats.pkl", "wb") as f:
    pickle.dump(stats, f)