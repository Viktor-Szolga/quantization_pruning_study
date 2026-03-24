import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
import os
from tqdm import tqdm

# --- Configuration & Paths ---# Assuming script is at same level as your pre-processing script
DATASET = "beauty"
OUT_DIR = f"processed_{DATASET}"
DATA_PATH = Path(os.path.join("data", OUT_DIR))

# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 20
LR = 0.001
MF_DIM = 8
LAYERS = [64, 32, 16, 8]
TRAIN_NEGATIVES = 4
TEST_NEGATIVES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NeuMF Model Architecture ---
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=8, layers=[64, 32, 16, 8]):
        super(NeuMF, self).__init__()
        # GMF Embeddings
        self.embedding_user_mf = nn.Embedding(num_embeddings=num_users + 1, embedding_dim=mf_dim)
        self.embedding_item_mf = nn.Embedding(num_embeddings=num_items + 1, embedding_dim=mf_dim)
        
        # MLP Embeddings
        self.embedding_user_mlp = nn.Embedding(num_embeddings=num_users + 1, embedding_dim=layers[0]//2)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=num_items + 1, embedding_dim=layers[0]//2)
        
        # MLP Layers
        self.fc_layers = nn.Sequential()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.add_module(f"linear_{idx}", nn.Linear(in_size, out_size))
            self.fc_layers.add_module(f"relu_{idx}", nn.ReLU())
            
        # Final Output Layer
        self.affine_output = nn.Linear(in_features=layers[-1] + mf_dim, out_features=1)

    def forward(self, user_indices, item_indices):
        # GMF Path
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        
        # MLP Path
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mlp_vector = self.fc_layers(mlp_vector)
        
        # Concatenate & Output (using BCEWithLogitsLoss so no Sigmoid here)
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        return logits.squeeze(-1)

# --- Dataset for Training (On-the-fly Uniform Negative Sampling) ---
class NMFDataset(Dataset):
    def __init__(self, data, user_item_set, num_items, num_negatives=4):
        self.users = data[:, 0]
        self.items = data[:, 1]
        self.user_item_set = user_item_set
        self.num_items = num_items
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        i = self.items[idx]
        
        neg_items = []
        for _ in range(self.num_negatives):
            while True:
                neg = np.random.randint(1, self.num_items + 1)
                if neg not in self.user_item_set[u]:
                    neg_items.append(neg)
                    break
                    
        users = torch.tensor([u] * (1 + self.num_negatives), dtype=torch.long)
        items = torch.tensor([i] + neg_items, dtype=torch.long)
        labels = torch.tensor([1.0] + [0.0] * self.num_negatives, dtype=torch.float)
        
        return users, items, labels

def collate_fn(batch):
    # Flatten batches since each item in the batch yields multiple user/item/label pairs
    users = torch.cat([item[0] for item in batch])
    items = torch.cat([item[1] for item in batch])
    labels = torch.cat([item[2] for item in batch])
    return users, items, labels

# --- Popularity Based Negative Sampling Function ---
def sample_negatives_pop(user, user_item_set, pop_prob, num_items, num_negs=100):
    negs = set()
    while len(negs) < num_negs:
        # Sample in bulk to speed up the loop
        batch = np.random.choice(num_items + 1, size=num_negs * 2, p=pop_prob)
        for neg in batch:
            if neg != 0 and neg not in user_item_set[user]:
                negs.add(neg)
            if len(negs) == num_negs:
                break
    return list(negs)

# --- Evaluation Function ---
def evaluate(model, test_data, user_item_set, pop_prob, num_items, device, batch_size=512):
    model.eval()
    hits, ndcgs = [], []
    
    test_users = test_data[:, 0]
    test_items = test_data[:, 1]
    
    # Pre-sample a large pool of negatives to avoid repeated np.random.choice calls
    neg_pool = np.random.choice(num_items + 1, size=len(test_users) * TEST_NEGATIVES * 2, p=pop_prob)
    pool_idx = 0

    for i in tqdm(range(0, len(test_users), batch_size), desc="Evaluating", leave=False):
        batch_end = min(i + batch_size, len(test_users))
        curr_batch_size = batch_end - i
        
        u_batch = test_users[i:batch_end]
        pos_i_batch = test_items[i:batch_end]
        
        # Build 101 items for each user in batch (1 pos + 100 negs)
        all_test_items = []
        for idx, u in enumerate(u_batch):
            user_negs = []
            while len(user_negs) < TEST_NEGATIVES:
                cand = neg_pool[pool_idx]
                pool_idx += 1
                if cand != 0 and cand not in user_item_set[u] and cand != pos_i_batch[idx]:
                    user_negs.append(cand)
                # Refill pool if empty
                if pool_idx >= len(neg_pool):
                    neg_pool = np.random.choice(num_items + 1, size=len(test_users) * 10, p=pop_prob)
                    pool_idx = 0
            all_test_items.append([pos_i_batch[idx]] + user_negs)
        
        # Convert to tensors: Shape [Batch, 101]
        item_tensor = torch.tensor(all_test_items, dtype=torch.long).to(device)
        user_tensor = torch.tensor(u_batch, dtype=torch.long).to(device).unsqueeze(1).expand(-1, 101)
        
        # Flatten for model: [Batch * 101]
        with torch.no_grad():
            logits = model(user_tensor.reshape(-1), item_tensor.reshape(-1))
            scores = logits.view(curr_batch_size, 101) # Reshape back to [Batch, 101]
            
        # The positive item is at index 0. Get its rank.
        # We find how many items have a score higher than the positive item.
        pos_scores = scores[:, 0].unsqueeze(1)
        ranks = (scores > pos_scores).sum(dim=1) 
        
        for r in ranks:
            if r < 10:
                hits.append(1)
                ndcgs.append(np.log(2) / np.log(r.item() + 2))
            else:
                hits.append(0)
                ndcgs.append(0)
                
    return np.mean(hits), np.mean(ndcgs)
# --- Main Execution Loop ---
if __name__ == "__main__":
    print("Loading data...")
    # Load Stats & Popularity
    with open(DATA_PATH / "stats.pkl", "rb") as f:
        num_users, num_items = pickle.load(f)
        
    with open(DATA_PATH / "popularity.pkl", "rb") as f:
        pop_data = pickle.load(f)
        pop_prob = pop_data["prob"]
        user_item_set = pop_data["user_item_set"]
        
    # Load NMF splits
    with open(DATA_PATH / "nmf" / "train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(DATA_PATH / "nmf" / "valid.pkl", "rb") as f:
        test_data = pickle.load(f)

    # Setup Dataset & DataLoader
    train_dataset = NMFDataset(train_data, user_item_set, num_items, num_negatives=TRAIN_NEGATIVES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Initialize Model, Loss, Optimizer
    model = NeuMF(num_users, num_items, mf_dim=MF_DIM, layers=LAYERS).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training Loop
    print(f"Starting Training on {DEVICE}...")
    best_hr = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for users, items, labels in pbar:
            users, items, labels = users.to(DEVICE), items.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(users, items)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        
        # Test Evaluation
        hr, ndcg = evaluate(model, test_data, user_item_set, pop_prob, num_items, DEVICE)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | HR@10: {hr:.4f} | NDCG@10: {ndcg:.4f}")
        
        if hr > best_hr:
            best_hr = hr
            torch.save(model.state_dict(), DATA_PATH / "nmf" / "best_neumf_model.pth")
            print("--> New best model saved!")
            
    print("Training complete.")