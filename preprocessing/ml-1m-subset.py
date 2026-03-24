from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm
import os
import numpy as np

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = "ml-1m-subset"
SEP = "::"
DATA_DIR = BASE_DIR / "data" / "ml-1m"

# Generate Structure
out_dir = "processed_ml-1m-subset"
os.makedirs(BASE_DIR / "data" / out_dir, exist_ok=True)
os.makedirs(BASE_DIR / "data" / out_dir / "nmf", exist_ok=True)
os.makedirs(BASE_DIR / "data" / out_dir / "bert", exist_ok=True)

# Load Data
full_ratings = pd.read_csv(
    DATA_DIR / "ratings.dat",
    sep=SEP,
    header=None,
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"],
    encoding="latin-1"
)    

# -------------------Subset Selection-----------------------
# Pick 5 users with at least 10 interactions
user_counts = full_ratings["UserID"].value_counts()
selected_users = user_counts[user_counts >= 10].index[:5]

subset_ratings = full_ratings[full_ratings["UserID"].isin(selected_users)].copy()

# For each user, keep only first 10 interactions (sorted by timestamp)
subset_ratings = subset_ratings.sort_values(by=["UserID", "Timestamp"], ascending=True)
subset_ratings = subset_ratings.groupby("UserID").head(10)

# Remap UserIDs to 1..5
user2id = {uid: idx + 1 for idx, uid in enumerate(sorted(selected_users))}
subset_ratings["UserID"] = subset_ratings["UserID"].map(user2id)

# Remap MovieIDs to 1..N (only items in subset)
unique_items = sorted(subset_ratings["MovieID"].unique())
item2id = {item: idx + 1 for idx, item in enumerate(unique_items)}
subset_ratings["MovieID"] = subset_ratings["MovieID"].map(item2id)

# -------------------NeuMF-----------------------
# Leave-One-Out split
sorted_ratings = subset_ratings.sort_values(by=["UserID", "Timestamp"], ascending=True)
item_rank = sorted_ratings.groupby("UserID").cumcount(ascending=False)

test_df = sorted_ratings[item_rank == 0].copy()
valid_df = sorted_ratings[item_rank == 1].copy()
train_df = sorted_ratings[item_rank >= 2].copy()

num_users = subset_ratings["UserID"].max()
num_items = len(unique_items)
stats = (num_users, num_items)

# Item popularity
item_counts = train_df["MovieID"].value_counts()
popularity = np.zeros(num_items + 1, dtype=np.float64)  # index 0 = padding
for item_id, count in item_counts.items():
    popularity[item_id] = count

popularity_smooth = popularity / popularity.sum()

with open(BASE_DIR / "data" / out_dir / "popularity.pkl", "wb") as f:
    pickle.dump({
        "counts": popularity,
        "prob": popularity_smooth
    }, f)

# Save NMF data
nmf_train = train_df[["UserID", "MovieID", "Rating"]].to_numpy()
nmf_valid = valid_df[["UserID", "MovieID", "Rating"]].to_numpy()
nmf_test  = test_df[["UserID", "MovieID", "Rating"]].to_numpy()

with open(BASE_DIR / "data" / out_dir / "nmf" / "train.pkl", "wb") as f:
    pickle.dump(nmf_train, f)
with open(BASE_DIR / "data" / out_dir / "nmf" / "valid.pkl", "wb") as f:
    pickle.dump(nmf_valid, f)
with open(BASE_DIR / "data" / out_dir / "nmf" / "test.pkl", "wb") as f:
    pickle.dump(nmf_test, f)
with open(BASE_DIR / "data" / out_dir / "stats.pkl", "wb") as f:
    pickle.dump(stats, f)

# -------------------------Bert4Rec--------------------------------
user_history = sorted_ratings.groupby("UserID")["MovieID"].apply(list).to_dict()

bert_train_sequences = {}
bert_valid_sequences = {}
bert_test_sequences = {}

train_user_ids = []
valid_user_ids = []
test_user_ids = []

for user_id, items in tqdm(user_history.items(), desc="Splitting", total=len(user_history)):
    if len(items) < 5:
        continue
    
    uid = int(user_id) - 1

    # Train
    bert_train_sequences[uid] = {"seq": items[:-2]}
    train_user_ids.append(uid)
    
    # Valid
    bert_valid_sequences[uid] = {"seq": items[:-2], "target": items[-2]}
    valid_user_ids.append(uid)
    
    # Test
    bert_test_sequences[uid] = {"seq": items[:-1], "target": items[-1]}
    test_user_ids.append(uid)

# Save Bert4Rec data
with open(BASE_DIR / "data" / out_dir / "bert" / "train.pkl", "wb") as f:
    pickle.dump((bert_train_sequences, train_user_ids), f)
with open(BASE_DIR / "data" / out_dir / "bert" / "valid.pkl", "wb") as f:
    pickle.dump((bert_valid_sequences, valid_user_ids), f)
with open(BASE_DIR / "data" / out_dir / "bert" / "test.pkl", "wb") as f:
    pickle.dump((bert_test_sequences, test_user_ids), f)
with open(BASE_DIR / "data" / out_dir / "stats.pkl", "wb") as f:
    pickle.dump(stats, f)