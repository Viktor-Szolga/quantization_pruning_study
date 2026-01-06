from pathlib import Path
import pandas as pd
import numpy as np
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "ml-1m"
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
full_ratings = pd.read_csv(
    DATA_DIR / "ratings.dat",
    sep="::",
    header=None,
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"],
    encoding="latin-1"
)

# LOO
sorted_ratings = full_ratings.sort_values(by=["UserID", "Timestamp"], ascending=True)

item_rank = sorted_ratings.groupby("UserID").cumcount(ascending=False)
test_df = sorted_ratings[item_rank == 0].copy()
valid_df = sorted_ratings[item_rank == 1].copy()
train_df = sorted_ratings[item_rank >= 2].copy()


nmf_train = train_df[["UserID", "MovieID", "Rating"]].to_numpy()
with open(BASE_DIR / "data" / "processed" / "nmf" / "train.pkl", "wb") as f:
    pickle.dump(nmf_train, f)
nmf_valid = valid_df[["UserID", "MovieID", "Rating"]].to_numpy()
with open(BASE_DIR / "data" / "processed" / "nmf" / "valid.pkl", "wb") as f:
    pickle.dump(nmf_valid, f)
nmf_test  = test_df[["UserID", "MovieID", "Rating"]].to_numpy()
with open(BASE_DIR / "data" / "processed" / "nmf" / "test.pkl", "wb") as f:
    pickle.dump(nmf_test, f)




user_history = sorted_ratings.groupby("UserID")["MovieID"].apply(list).to_dict()

bert_train_sequences = {}
bert_valid_sequences = {}
bert_test_sequences = {}

for user_id, items in user_history.items():
    if len(items) < 3: # Need at least 3 items for Train/Valid/Test LOO
        continue
        
    # Train: All items except the last two
    bert_train_sequences[user_id-1] = {
        "seq": items[:-2]
    }
    
    # Valid: Input is train items, Target is the second to last item
    # We save both so the Dataset knows what to mask and what to check
    bert_valid_sequences[user_id-1] = {
        "seq": items[:-2], 
        "target": items[-2]
    }
    
    # Test: Input is train + valid items, Target is the very last item
    bert_test_sequences[user_id-1] = {
        "seq": items[:-1], 
        "target": items[-1]
    }
#bert_train_sequences = train_df.groupby("UserID")["MovieID"].apply(list).to_dict()
with open(BASE_DIR / "data" / "processed" / "bert" / "train.pkl", "wb") as f:
    pickle.dump(bert_train_sequences, f)
#bert_valid_targets = valid_df.set_index("UserID")["MovieID"].to_dict()
with open(BASE_DIR / "data" / "processed" / "bert" / "valid.pkl", "wb") as f:
    pickle.dump(bert_valid_sequences, f)
#bert_test_targets  = test_df.set_index("UserID")["MovieID"].to_dict()
with open(BASE_DIR / "data" / "processed" / "bert" / "test.pkl", "wb") as f:
    pickle.dump(bert_test_sequences, f)



