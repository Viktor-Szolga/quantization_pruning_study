import torch
import numpy as np
from src.models import NeuralMF
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
        
    def train_epoch_nmf(self, loader, num_items):
        self.model.train()
        total_loss = 0

        for batch in loader:
            users, pos_items, _ = [b.to(self.device) for b in batch]
            batch_size = users.size(0)

            neg_items = torch.randint(0, num_items, (batch_size,)).to(self.device)

            all_users = torch.cat([users, users], dim=0)
            all_items = torch.cat([pos_items, neg_items], dim=0)
            
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
    
    def train_n_steps_bert(self, train_loader, validation_loader, max_steps, validation_interval=1000, k=10):
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

            tokens, labels = [b.to(self.device) for b in batch]

            self.optimizer.zero_grad()

            logits = self.model(tokens)
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
                    save_path = Path("trained_models") / f"best_bert_model.pth"
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
            hr_list = []
            ndcg_list = []

            with torch.no_grad():
                for batch in loader:
                    seq, items = [b.to(self.device) for b in batch]

                    batch_size, num_candidates = items.shape
                    current_k = min(k, num_candidates)

                    logits = self.model(seq)
                    scores_full = logits[:, -1, :]

                    scores = scores_full.gather(1, items)
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
    
    def report_model_size(self):
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f"Model Size: {size_all_mb:.2f} MB")
        return size_all_mb