from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm
import os

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = "ml-1m"
SEP = "::"
DATA_DIR = BASE_DIR / "data" / "ml-1m"

# Generate Structure
out_dir = "processed_ml-1m"
os.makedirs(BASE_DIR / "data" / out_dir, exist_ok=True)
os.makedirs(BASE_DIR / "data" / out_dir / "nmf", exist_ok=True)
os.makedirs(BASE_DIR / "data" / out_dir/ "bert", exist_ok=True)

# Load Data
full_ratings = pd.read_csv(
    DATA_DIR / "ratings.dat",
    sep=SEP,
    header=None,
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"],
    encoding="latin-1"
)    
print(full_ratings.head())

# -------------------NeuMF-----------------------
# LOO
sorted_ratings = full_ratings.sort_values(by=["UserID", "Timestamp"], ascending=True)

item_rank = sorted_ratings.groupby("UserID").cumcount(ascending=False)
test_df = sorted_ratings[item_rank == 0].copy()
valid_df = sorted_ratings[item_rank == 1].copy()
train_df = sorted_ratings[item_rank >= 2].copy()


num_users = full_ratings["UserID"].max()
num_items = full_ratings["MovieID"].max()

stats = (num_users, num_items)

# Save data
nmf_train = train_df[["UserID", "MovieID", "Rating"]].to_numpy()
with open(BASE_DIR / "data" / out_dir / "nmf" / "train.pkl", "wb") as f:
    pickle.dump(nmf_train, f)
nmf_valid = valid_df[["UserID", "MovieID", "Rating"]].to_numpy()
with open(BASE_DIR / "data" / out_dir / "nmf" / "valid.pkl", "wb") as f:
    pickle.dump(nmf_valid, f)
nmf_test  = test_df[["UserID", "MovieID", "Rating"]].to_numpy()
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
    # Users need at least 3 interactions
    if len(items) < 3:
        continue
    
    # Bert uses zero based user ids be careful when remapping to users during data analysis
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



# Save data
with open(BASE_DIR / "data" / out_dir / "bert" / "train.pkl", "wb") as f:
    pickle.dump((bert_train_sequences, train_user_ids), f)

with open(BASE_DIR / "data" / out_dir / "bert" / "valid.pkl", "wb") as f:
    pickle.dump((bert_valid_sequences, valid_user_ids), f)

with open(BASE_DIR / "data" / out_dir / "bert" / "test.pkl", "wb") as f:
    pickle.dump((bert_test_sequences, test_user_ids), f)

with open(BASE_DIR / "data" / out_dir / "stats.pkl", "wb") as f:
    pickle.dump(stats, f)

