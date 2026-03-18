from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os

BASE_DIR = Path(__file__).resolve().parent.parent
dataset = "ml-1m"
sep = "," if dataset == "ml-20m" else "::"
DATA_DIR = BASE_DIR / "data" / dataset
"""
full_users = pd.read_csv(
    DATA_DIR / "users.dat",
    sep="::",
    header=None,
    engine="python",
    names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
    encoding="latin-1"
)

full_movies = pd.read_csv(
    DATA_DIR / "movies.dat",
    sep="::",
    header=None,
    engine="python",
    names=["MovieID", "Title", "Genres"],
    encoding="latin-1"
)
"""
if dataset == "ml-1m":
    full_ratings = pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep=sep,
        header=None,
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1"
    )
if dataset == "ml-20m":
    full_ratings = pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep=sep,
        engine="python",
        header=0,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1"
    )
print(full_ratings["MovieID"].max())
print(full_ratings["UserID"].max())

# LOO
sorted_ratings = full_ratings.sort_values(by=["UserID", "Timestamp"], ascending=True)

item_rank = sorted_ratings.groupby("UserID").cumcount(ascending=False)
test_df = sorted_ratings[item_rank == 0].copy()
valid_df = sorted_ratings[item_rank == 1].copy()
train_df = sorted_ratings[item_rank >= 2].copy()


out_dir = "processed_ml-1m" if dataset == "ml-1m" else "processed20"
os.makedirs(BASE_DIR / "data" / out_dir, exist_ok=True)
os.makedirs(BASE_DIR / "data" / out_dir / "nmf", exist_ok=True)
os.makedirs(BASE_DIR / "data" / out_dir/ "bert", exist_ok=True)

nmf_train = train_df[["UserID", "MovieID", "Rating"]].to_numpy()
with open(BASE_DIR / "data" / out_dir / "nmf" / "train.pkl", "wb") as f:
    pickle.dump(nmf_train, f)
nmf_valid = valid_df[["UserID", "MovieID", "Rating"]].to_numpy()
with open(BASE_DIR / "data" / out_dir / "nmf" / "valid.pkl", "wb") as f:
    pickle.dump(nmf_valid, f)
nmf_test  = test_df[["UserID", "MovieID", "Rating"]].to_numpy()
with open(BASE_DIR / "data" / out_dir / "nmf" / "test.pkl", "wb") as f:
    pickle.dump(nmf_test, f)




user_history = sorted_ratings.groupby("UserID")["MovieID"].apply(list).to_dict()

bert_train_sequences = {}
bert_valid_sequences = {}
bert_test_sequences = {}

train_user_ids = []
valid_user_ids = []
test_user_ids = []

for user_id, items in tqdm(user_history.items(), desc="Splitting", total=len(user_history)):
    if len(items) < 3:
        continue
        
    uid = int(user_id) - 1

    # Train
    bert_train_sequences[uid] = {
        "seq": items[:-2]
    }
    train_user_ids.append(uid)
    
    # Valid
    bert_valid_sequences[uid] = {
        "seq": items[:-2], 
        "target": items[-2]
    }
    valid_user_ids.append(uid)
    
    # Test
    bert_test_sequences[uid] = {
        "seq": items[:-1], 
        "target": items[-1]
    }
    test_user_ids.append(uid)


with open(BASE_DIR / "data" / out_dir / "bert" / "train.pkl", "wb") as f:
    pickle.dump((bert_train_sequences, train_user_ids), f)

with open(BASE_DIR / "data" / out_dir / "bert" / "valid.pkl", "wb") as f:
    pickle.dump((bert_valid_sequences, valid_user_ids), f)

with open(BASE_DIR / "data" / out_dir / "bert" / "test.pkl", "wb") as f:
    pickle.dump((bert_test_sequences, test_user_ids), f)

