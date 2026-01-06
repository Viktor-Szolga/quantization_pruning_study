import torch
import numpy as np
from sklearn.metrics import ndcg_score

class RecSysTrainer:
    def __init__(self, model, optimizer, criterion, device="cuda" if torch.cuda.is_available else "cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device


    def train_epoch(self, loader, num_items):
        self.model.train()
        total_loss = 0

        for batch in loader:
            # 1. Unpack the real data (Positives)
            # We ignore 'labels' (the 1-5 rating) because we treat all as 1.0
            users, pos_items, _ = [b.to(self.device) for b in batch]
            batch_size = users.size(0)

            # 2. Generate Negative Samples
            # Pick a random item for every user in the batch
            # Labels for these will be 0.0
            neg_items = torch.randint(0, num_items, (batch_size,)).to(self.device)

            # 3. Combine Positives and Negatives for a single forward pass
            # This is more efficient than two separate passes
            all_users = torch.cat([users, users], dim=0)
            all_items = torch.cat([pos_items, neg_items], dim=0)
            
            # Create target labels: 1s for the first half, 0s for the second
            pos_labels = torch.ones(batch_size).to(self.device)
            neg_labels = torch.zeros(batch_size).to(self.device)
            all_labels = torch.cat([pos_labels, neg_labels], dim=0)

            # 4. Standard Training Step
            self.optimizer.zero_grad()
            preds = self.model(all_users, all_items)
            
            # Ensure BCEWithLogitsLoss is used in the constructor
            loss = self.criterion(preds, all_labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)
    
    def evaluate(self, loader, k=10):
        self.model.eval()
        ndcg_list = []
        hr_list = []

        with torch.no_grad():
            for batch in loader:
                users, items, labels = [b.to(self.device) for b in batch]
                batch_size, num_candidate_items = items.shape

                users_flat = users.repeat_interleave(num_candidate_items)
                items_flat = items.view(-1)

                scores = self.model(users_flat, items_flat)
                scores = scores.view(batch_size, num_candidate_items)

                _, indices = torch.topk(scores, k=k, dim=1)

                for i in range(batch_size):
                    if 0 in indices[i]:
                        hr_list.append(1.0)
                        rank = (indices[i] == 0).nonzero(as_tuple=True)[0].item()
                        ndcg_list.append(1.0 / np.log2(rank + 2))
                    else:
                        hr_list.append(0.0)
                        ndcg_list.append(0.0)
                
                
        return np.mean(hr_list), np.mean(ndcg_list)
    
    def report_model_size(self):
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f"Model Size: {size_all_mb:.2f} MB")
        return size_all_mb