import torch
import numpy as np
from sklearn.metrics import ndcg_score
from src.models import Bert4Rec, NeuralMF
from tqdm import tqdm
from pathlib import Path

class RecSysTrainer:
    def __init__(self, model, optimizer, criterion, device="cpu", scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

    def load_state_dict(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        
    def train_epoch(self, loader, num_items):
        if isinstance(self.model, NeuralMF):
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

                self.optimizer.zero_grad()
                preds = self.model(all_users, all_items)
                
                loss = self.criterion(preds.squeeze(), all_labels.squeeze())
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            return total_loss / len(loader)
        
        if isinstance(self.model, Bert4Rec):
            self.model.train()
            total_loss = 0

            for batch in loader:
                # tokens: the sequence with [MASK] tokens (shape: batch_size, max_len)
                # labels: the ground truth item IDs only for the masked positions, else 0
                tokens, labels = [b.to(self.device) for b in batch]

                self.optimizer.zero_grad()
                
                # Forward pass: logits shape [batch_size, max_len, num_items + 2]
                logits = self.model(tokens)
                
                # We only want to calculate loss for the positions that were actually masked.
                # labels == 0 are unmasked positions or padding.
                # CrossEntropyLoss 'ignore_index=0' handles this automatically.
                
                # Reshape for CrossEntropyLoss: (N, C, L) or flatten to (N*L, C)
                # logits.view(-1, logits.size(-1)) -> [batch_size * max_len, num_items + 2]
                # labels.view(-1) -> [batch_size * max_len]
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                
                total_loss += loss.item()
                
            return total_loss / len(loader)
    
    def train_n_steps(self, train_loader, validation_loader, max_steps, validation_interval=1000, k=10):
        if isinstance(self.model, Bert4Rec):
            self.model.train()
            train_losses = []
            val_hr = []
            val_ndcg = []
            eval_at = []
            train_iter = iter(train_loader)
            best_ndcg = float("-inf")
            for i in tqdm(range(max_steps), desc="Training", total=max_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                # tokens: the sequence with [MASK] tokens (shape: batch_size, max_len)
                # labels: the ground truth item IDs only for the masked positions, else 0
                tokens, labels = [b.to(self.device) for b in batch]

                self.optimizer.zero_grad()
                
                # Forward pass: logits shape [batch_size, max_len, num_items + 2]
                logits = self.model(tokens)
                
                # We only want to calculate loss for the positions that were actually masked.
                # labels == 0 are unmasked positions or padding.
                # CrossEntropyLoss 'ignore_index=0' handles this automatically.
                
                # Reshape for CrossEntropyLoss: (N, C, L) or flatten to (N*L, C)
                # logits.view(-1, logits.size(-1)) -> [batch_size * max_len, num_items + 2]
                # labels.view(-1) -> [batch_size * max_len]
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                
                train_losses.append(loss.item())

                if i > 0 and i % validation_interval == 0:
                    hr, ndcg = self.evaluate(loader=validation_loader)
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        save_path = Path("trained_models") / f"best_bert_model_num_steps.pth"
                        torch.save(self.model.state_dict(), str(save_path))
                    val_hr.append(hr)
                    val_ndcg.append(ndcg)
                    eval_at.append(i)
                    self.model.train()
                
            return train_losses, val_hr, val_ndcg, eval_at
        
    def evaluate(self, loader, k=10):
        if isinstance(self.model, NeuralMF):
            self.model.eval()
            ndcg_list = []
            hr_list = []

            with torch.no_grad():
                for batch in loader:
                    users, items, labels = [b.to(self.device) for b in batch]
                
                    if items.dim() == 1:
                        items = items.unsqueeze(1)
                    
                    batch_size, num_candidate_items = items.shape

                    current_k = min(k, num_candidate_items)

                    users_flat = users.repeat_interleave(num_candidate_items)
                    items_flat = items.reshape(-1)

                    scores = self.model(users_flat, items_flat)
                    scores = scores.view(batch_size, num_candidate_items)

                    _, indices = torch.topk(scores, k=current_k, dim=1)

                    for i in range(batch_size):
                        if 0 in indices[i]: 
                            hr_list.append(1.0)
                            rank = (indices[i] == 0).nonzero(as_tuple=True)[0].item()
                            ndcg_list.append(1.0 / np.log2(rank + 2))
                        else:
                            hr_list.append(0.0)
                            ndcg_list.append(0.0)
                    
            return np.mean(hr_list), np.mean(ndcg_list)
        
        else:
            self.model.eval()
            hit_rate = []
            ndcg = []
            
            with torch.no_grad():
                for batch in loader:
                    # The BERTDataset returns (tokens, target)
                    # seq: [batch_size, max_len]
                    # target: [batch_size, 1]
                    seq, target = batch
                    seq = seq.to(self.device)
                    target = target.to(self.device).squeeze() # Flatten to [batch_size]

                    # 1. Forward Pass
                    # logits shape: [batch_size, seq_len, vocab_size]
                    logits = self.model(seq)
                    
                    # 2. Extract prediction for the [MASK] token (the very last position)
                    # In validation, our dataset ensures the last token is always [MASK]
                    mask_logits = logits[:, -1, :] 
                    
                    # 3. Calculate Metrics (Top-K)
                    _, top_indices = torch.topk(mask_logits, k, dim=-1)
                    
                    # Move to CPU for metric calculation
                    top_indices = top_indices.cpu().numpy()
                    target = target.cpu().numpy()

                    for i in range(len(target)):
                        true_item = target[i]
                        top_k_items = top_indices[i]
                        
                        # Hit Rate @ K
                        if true_item in top_k_items:
                            hit_rate.append(1)
                            
                            # NDCG @ K
                            # find index returns the rank (0-indexed)
                            rank = np.where(top_k_items == true_item)[0][0]
                            ndcg.append(1 / np.log2(rank + 2))
                        else:
                            hit_rate.append(0)
                            ndcg.append(0)
                    #return 0, 0
            return np.mean(hit_rate), np.mean(ndcg)
    
    def report_model_size(self):
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f"Model Size: {size_all_mb:.2f} MB")
        return size_all_mb