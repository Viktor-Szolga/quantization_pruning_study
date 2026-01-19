from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm

# =====================
# CONFIG
# =====================
dataset = "ml-20m"   # "ml-1m" or "ml-20m"
MIN_INTERACTIONS = 5

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / dataset
OUT_DIR = BASE_DIR / "data" / "processed20" if dataset == "ml-20m" else BASE_DIR / "data" / "processed1m"

(OUT_DIR / "nmf").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "bert").mkdir(parents=True, exist_ok=True)

sep = "," if dataset == "ml-20m" else "::"

# =====================
# LOAD RATINGS
# =====================
if dataset == "ml-1m":
    full_ratings = pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep=sep,
        header=None,
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1"
    )
else:  # ml-20m
    full_ratings = pd.read_csv(
        DATA_DIR / "ratings.csv",
        sep=sep,
        header=0,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1"
    )

print("Original stats:")
print("Users:", full_ratings["UserID"].nunique())
print("Items:", full_ratings["MovieID"].nunique())
print("Interactions:", len(full_ratings))
print()

# =====================
# ITERATIVE FILTERING
# =====================
def filter_min_interactions(df, min_user=5, min_item=5):
    while True:
        before = len(df)

        user_counts = df["UserID"].value_counts()
        df = df[df["UserID"].isin(user_counts[user_counts >= min_user].index)]

        item_counts = df["MovieID"].value_counts()
        df = df[df["MovieID"].isin(item_counts[item_counts >= min_item].index)]

        if len(df) == before:
            break
    return df

full_ratings = filter_min_interactions(
    full_ratings,
    min_user=MIN_INTERACTIONS,
    min_item=MIN_INTERACTIONS
)

print(f"After filtering (<{MIN_INTERACTIONS} removed):")
print("Users:", full_ratings["UserID"].nunique())
print("Items:", full_ratings["MovieID"].nunique())
print("Interactions:", len(full_ratings))
print()

# =====================
# REMAP IDS (START AT 1, NO GAPS)
# =====================
user_map = {u: i + 1 for i, u in enumerate(full_ratings["UserID"].unique())}
item_map = {m: i + 1 for i, m in enumerate(full_ratings["MovieID"].unique())}

full_ratings["UserID"] = full_ratings["UserID"].map(user_map)
full_ratings["MovieID"] = full_ratings["MovieID"].map(item_map)

print("After remapping:")
print("UserID range:", full_ratings["UserID"].min(), "-", full_ratings["UserID"].max())
print("MovieID range:", full_ratings["MovieID"].min(), "-", full_ratings["MovieID"].max())
print()

# =====================
# SORT + LOO SPLIT
# =====================
sorted_ratings = full_ratings.sort_values(by=["UserID", "Timestamp"])

item_rank = sorted_ratings.groupby("UserID").cumcount(ascending=False)

test_df = sorted_ratings[item_rank == 0].copy()
valid_df = sorted_ratings[item_rank == 1].copy()
train_df = sorted_ratings[item_rank >= 2].copy()

# =====================
# SAVE NMF DATA
# =====================
nmf_train = train_df[["UserID", "MovieID", "Rating"]].to_numpy()
nmf_valid = valid_df[["UserID", "MovieID", "Rating"]].to_numpy()
nmf_test  = test_df[["UserID", "MovieID", "Rating"]].to_numpy()

with open(OUT_DIR / "nmf" / "train.pkl", "wb") as f:
    pickle.dump(nmf_train, f)
with open(OUT_DIR / "nmf" / "valid.pkl", "wb") as f:
    pickle.dump(nmf_valid, f)
with open(OUT_DIR / "nmf" / "test.pkl", "wb") as f:
    pickle.dump(nmf_test, f)

# =====================
# BUILD BERT SEQUENCES
# =====================
user_history = sorted_ratings.groupby("UserID")["MovieID"].apply(list).to_dict()

bert_train = {}
bert_valid = {}
bert_test = {}

for user_id, items in tqdm(user_history.items(), desc="Building BERT sequences"):
    if len(items) < 3:
        continue

    bert_train[user_id-1] = {"seq": items[:-2]}
    bert_valid[user_id-1] = {"seq": items[:-2], "target": items[-2]}
    bert_test[user_id-1]  = {"seq": items[:-1], "target": items[-1]}

with open(OUT_DIR / "bert" / "train.pkl", "wb") as f:
    pickle.dump(bert_train, f)
with open(OUT_DIR / "bert" / "valid.pkl", "wb") as f:
    pickle.dump(bert_valid, f)
with open(OUT_DIR / "bert" / "test.pkl", "wb") as f:
    pickle.dump(bert_test, f)

# =====================
# FINAL SUMMARY
# =====================
print("Preprocessing complete ✅")
print(f"Saved to: {OUT_DIR}")
print("Final users:", len(bert_train))
print("Final items:", full_ratings['MovieID'].nunique())
