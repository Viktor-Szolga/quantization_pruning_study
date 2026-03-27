import torch
import numpy as np
from src.models import NeuMF
from tqdm import tqdm
from pathlib import Path
import os

class RecSysTrainer:
    def __init__(self, model, optimizer, criterion, device="cpu", scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

    def train_n_steps_nmf(self, train_loader, validation_loader, cfg, save_path):
        train_losses = []
        val_hr = []
        val_ndcg = []
        eval_at = []

        train_iter = iter(train_loader)
        best_ndcg = float("-inf")
        patience_counter = 0

        self.optimizer.zero_grad()
        self.model.train()
        for i in tqdm(range(cfg.training.max_steps), desc=f"{cfg.model.type} on {cfg.dataset.name}", total=cfg.training.max_steps, leave=False):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            users, items, _ = [b.to(self.device) for b in batch]
            pos_items = items[:, 0]
            neg_items = items[:, 1:]
            batch_size = users.size(0)
            K = neg_items.size(1)

            pos_preds = self.model(users, pos_items)
            neg_preds = self.model(
                users.repeat_interleave(K),
                neg_items.reshape(-1)
            ).view(batch_size, K)

            pos_labels = torch.ones_like(pos_preds)
            neg_labels = torch.zeros(batch_size * K, device=self.device)
            
            preds = torch.cat([pos_preds, neg_preds.reshape(-1)])
            labels = torch.cat([pos_labels, neg_labels])
            
            loss = self.criterion(preds, labels)
            loss / cfg.training.accumulation_steps
            loss.backward()
            if cfg.training.get("max_norm", None) is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.training.max_norm)

            if (i + 1) % cfg.training.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

            train_losses.append(loss.item())

            if (i + 1) % (cfg.evaluation.interval * cfg.training.accumulation_steps) == 0 or (i+1) == cfg.training.max_steps:
                hr, ndcg = self.evaluate(loader=validation_loader)

                if ndcg > best_ndcg and i > cfg.training.max_steps * cfg.training.get("warmup_ratio", 0):
                    best_ndcg = ndcg
                    torch.save(self.model.state_dict(), f"{save_path}.pth")
                    patience_counter = 0
                elif i > cfg.training.max_steps * cfg.training.get("warmup_ratio", 0):
                    patience_counter += 1

                val_hr.append(hr)
                val_ndcg.append(ndcg)
                eval_at.append(i + 1)

                self.model.train()

                if patience_counter >= cfg.training.early_stopping_patience:
                    print(f"Early stopping at step {i + 1}")
                    break
        
        if cfg.training.max_steps % cfg.training.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()
        return train_losses, val_hr, val_ndcg, eval_at

    def train_n_steps_bert(self, train_loader, validation_loader, accumulation_steps, max_steps, validation_interval=1000, k=10, save_path="bert", cfg=None):
        self.model.train()
        train_losses = []
        val_hr = []
        val_ndcg = []
        eval_at = []
        train_iter = iter(train_loader)
        best_ndcg = float("-inf")
        patience_counter = 0
        self.optimizer.zero_grad()
        for i in tqdm(range(max_steps), desc=f"{cfg.model.type} on {cfg.dataset.name}", total=max_steps, leave=False):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            tokens, labels, _ = [b.to(self.device) for b in batch]

            logits = self.model(tokens)
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            #-----------_Added---------------
            loss = loss.sum() / (labels != 0).sum()
            #-------------End Added---------------
            loss = loss/accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()
            
            train_losses.append(loss.item() * accumulation_steps)

            if (i + 1) % (cfg.evaluation.interval * accumulation_steps) == 0 or (i+1) == cfg.training.max_steps:
                hr, ndcg = self.evaluate(loader=validation_loader)
                if ndcg > best_ndcg and i > cfg.training.max_steps * cfg.training.get("warmup_ratio", 0):
                    best_ndcg = ndcg
                    torch.save(self.model.state_dict(), f"{save_path}.pth")
                    patience_counter = 0
                elif i > cfg.training.max_steps * cfg.training.get("warmup_ratio", 0):
                    patience_counter += 1

                val_hr.append(hr)
                val_ndcg.append(ndcg)
                eval_at.append(i+1)
                self.model.train()
                if patience_counter >= cfg.training.early_stopping_patience:
                    print(f"Early stopping at step {i + 1} (no improvement for {cfg.training.early_stopping_patience} evaluations)")
                    break

        if max_steps % accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()
        return train_losses, val_hr, val_ndcg, eval_at
        
    def evaluate(self, loader, k=10, performance_per_user=False):
        if performance_per_user:
            user_hr = {}
            user_ndcg = {}
        if isinstance(self.model, NeuMF):
            self.model.eval()
            ndcg_list = []
            hr_list = []

            with torch.no_grad():
                for batch in tqdm(loader, desc="Validating", total=len(loader), leave=False):
                    users, items, _ = [b.to(self.device) for b in batch]
                    if items.dim() == 1:
                        items = items.unsqueeze(1)
                    
                    batch_size, num_candidate_items = items.shape

                    current_k = min(k, num_candidate_items)

                    users_flat = users.repeat_interleave(num_candidate_items)
                    items_flat = items.reshape(-1)

                    scores = self.model(users_flat, items_flat)
                    scores = scores + torch.randn_like(scores) * 1e-6
                    scores = scores.view(batch_size, num_candidate_items)

                    _, indices = torch.topk(scores, k=current_k, dim=1)

                    for i in range(batch_size):
                        user_id = users[i].item()
                        #--------------Changed-------------
                        #if 0 in indices[i]: 
                        if (indices[i] == 0).any():
                            #-----------___End changed_---------------
                            hr = 1.0
                            rank = (indices[i] == 0).nonzero(as_tuple=True)[0].item()
                            ndcg = 1.0 / np.log2(rank + 2)
                        else:
                            hr = 0.0
                            ndcg = 0.0
                        if performance_per_user:
                            user_hr[user_id] = hr
                            user_ndcg[user_id] = ndcg
                        hr_list.append(hr)
                        ndcg_list.append(ndcg)
            if performance_per_user:
                return np.mean(hr_list), np.mean(ndcg_list), user_hr, user_ndcg
            return np.mean(hr_list), np.mean(ndcg_list)
        
        else:
            self.model.eval()
            hr_list = []
            ndcg_list = []

            with torch.no_grad():
                for batch in tqdm(loader, desc="Validating", total=len(loader), leave=False):
                    seq, items, users = [b.to(self.device) for b in batch]

                    batch_size, num_candidates = items.shape
                    current_k = min(k, num_candidates)

                    logits = self.model(seq)
                    scores_full = logits[:, -1, :]

                    scores = scores_full.gather(1, items)
                    scores = scores + torch.randn_like(scores) * 1e-6
                    _, indices = torch.topk(scores, k=current_k, dim=1)

                    for i in range(batch_size):
                        user_id = users[i].item()
                        if 0 in indices[i]: 
                            hr = 1.0
                            rank = (indices[i] == 0).nonzero(as_tuple=True)[0].item()
                            ndcg = 1.0 / np.log2(rank + 2)
                        else:
                            hr = 0.0
                            ndcg = 0.0
                        if performance_per_user:
                            user_hr[user_id] = hr
                            user_ndcg[user_id] = ndcg
                        hr_list.append(hr)
                        ndcg_list.append(ndcg)

            if performance_per_user:
                return np.mean(hr_list), np.mean(ndcg_list), user_hr, user_ndcg
            return np.mean(hr_list), np.mean(ndcg_list)

    def measure_metrics(self, loader):
        if isinstance(self.model, NeuMF):
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(loader, desc="Validating", total=len(loader), leave=False):
                    if hasattr(self.model, "large_test_emb"):
                        self.model.large_test_emb(torch.randint(0, 1_000_000, (256,), device=self.device))
                    users, items, _ = [b.to(self.device) for b in batch]
                    if items.dim() == 1:
                        items = items.unsqueeze(1)
                    self.model(users.repeat_interleave(items.shape[-1]), items.reshape(-1))
        else:
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(loader, desc="Validating", total=len(loader), leave=False):
                    if hasattr(self.model, "large_test_emb"):
                        self.model.large_test_emb(torch.randint(0, 1_000_000, (256,), device=self.device))
                    seq, _, _ = [b.to(self.device) for b in batch]
                    self.model(seq)

