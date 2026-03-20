import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
import pandas as pd
import time
import psutil
import threading
from omegaconf import OmegaConf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- CUSTOM IMPORTS ---
from src.data_manager import DataManager
from src.models import Bert4Rec
from src.trainer import RecSysTrainer
from src.utils import set_seed

# ==========================================
# 1. QUANTIZATION CLASSES (Memory-Safe)
# ==========================================
class CPU16bitAbsmaxEmbedding(nn.Module):
    def __init__(self, source_layer):
        super().__init__()
        self.num_embeddings = source_layer.num_embeddings
        self.embedding_dim = source_layer.embedding_dim
        
        X = source_layer.weight.data.detach().to(torch.float32)
        source_layer.weight.data = torch.empty(0) 
        
        self.register_buffer("absmax", X.abs().max() + 1e-8)
        self.register_buffer("c", 65504.0 / self.absmax)
        
        # Store as half to keep the "Inference RAM" low
        self.weight_quant = nn.Parameter((self.c * X).half())
        
        del X
        gc.collect()

    def forward(self, x):
        return F.embedding(x, self.weight_quant).float() / self.c

class CPU8bitAbsmaxEmbedding(nn.Module):
    def __init__(self, source_layer):
        super().__init__()
        self.num_embeddings = source_layer.num_embeddings
        self.embedding_dim = source_layer.embedding_dim
        
        X = source_layer.weight.data.detach().to(torch.float32)
        source_layer.weight.data = torch.empty(0)

        self.register_buffer("absmax", X.abs().max() + 1e-8)
        self.register_buffer("c", 127.0 / self.absmax)
        self.register_buffer("weight_quant", torch.round(self.c * X).to(torch.int8))
        
        del X
        gc.collect()

    def forward(self, x):
        return F.embedding(x, self.weight_quant).to(torch.float32) / self.c

class CPU4bitAbsmaxEmbedding(nn.Module):
    def __init__(self, source_layer):
        super().__init__()
        self.num_embeddings = source_layer.num_embeddings
        self.embedding_dim = source_layer.embedding_dim
        
        X = source_layer.weight.data.detach().to(torch.float32)
        source_layer.weight.data = torch.empty(0)

        self.register_buffer("absmax", X.abs().max() + 1e-8)
        self.register_buffer("c", 7.0 / self.absmax)
        
        # Quantize in one go (Linear is cheap)
        X_uint4 = (torch.round(self.c * X).clamp(-7, 7) + 7).to(torch.uint8)
        self.register_buffer("weight_quant_packed", (X_uint4[:, 0::2] << 4) | X_uint4[:, 1::2])
        
        del X, X_uint4
        gc.collect()

    def forward(self, x):
        p = F.embedding(x, self.weight_quant_packed)
        high, low = (p >> 4) & 0x0F, p & 0x0F
        X_int4 = torch.stack([high, low], dim=-1).reshape(*x.shape, self.embedding_dim).to(torch.float32)
        return (X_int4 - 7.0) / self.c

class CPUNF4Embedding(nn.Module):
    def __init__(self, source_layer):
        super().__init__()
        self.num_embeddings = source_layer.num_embeddings
        self.embedding_dim = source_layer.embedding_dim
        
        # Numbers taken from Q-LoRA paper appendix
        nf4_lut = [-1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, 
                   -0.09105, 0.0, 0.07958, 0.16093, 0.24611, 0.33791, 0.44071, 0.56261, 0.72295, 1.0]
        self.register_buffer("nf4_lut", torch.tensor(nf4_lut))

        X = source_layer.weight.data.detach().to(torch.float32)
        source_layer.weight.data = torch.empty(0)

        self.register_buffer("absmax", X.abs().max() + 1e-8)
        self.register_buffer("c", 1.0 / self.absmax)
        X_norm = self.c * X
        
        # MICRO-CHUNKING
        indices = torch.zeros((self.num_embeddings, self.embedding_dim), dtype=torch.uint8)
        chunk_size = 5000 
        
        for i in range(0, self.num_embeddings, chunk_size):
            end_i = min(i + chunk_size, self.num_embeddings)
            
            dist = (X_norm[i:end_i].unsqueeze(-1) - self.nf4_lut).abs()
            indices[i:end_i] = dist.argmin(dim=-1).to(torch.uint8)
            
            if i % 50000 == 0: gc.collect()

        self.register_buffer("weight_quant_packed", (indices[:, 0::2] << 4) | indices[:, 1::2])
        del X, X_norm, indices
        gc.collect()

    def forward(self, x):
        p = F.embedding(x, self.weight_quant_packed)
        idx = torch.stack([(p >> 4) & 0x0F, p & 0x0F], dim=-1).reshape(*x.shape, self.embedding_dim)
        return self.nf4_lut[idx.long()] / self.c
    
# ==========================================
# 2. UPDATED MAIN PIPELINE
# ==========================================

class MemoryMonitor:
    """ Background thread to track peak RAM usage of the current process. """
    def __init__(self):
        self.keep_running = True
        self.peak_ram = 0.0
        self.process = psutil.Process(os.getpid())

    def measure(self):
        while self.keep_running:
            current_ram = self.process.memory_info().rss / (1024 * 1024)
            if current_ram > self.peak_ram:
                self.peak_ram = current_ram
            time.sleep(0.001) # Poll every 1ms

def quant_embedding_size_mb(model):
    if isinstance(model.item_embedding, CPU16bitAbsmaxEmbedding): weight = model.item_embedding.weight_quant
    if isinstance(model.item_embedding, CPU8bitAbsmaxEmbedding): weight = model.item_embedding.weight_quant
    if isinstance(model.item_embedding, CPU4bitAbsmaxEmbedding): weight = model.item_embedding.weight_quant_packed
    if isinstance(model.item_embedding, CPUNF4Embedding): weight = model.item_embedding.weight_quant_packed
    if isinstance(model.item_embedding, nn.Embedding): weight = model.item_embedding.weight
    return weight.element_size() * weight.numel() / 1024**2

def test_for_normal(model, dataset):
    w = model.item_embedding.weight.detach().cpu().numpy()
    w_no0 = w[1:, :]   # skip first row directly
    w = w_no0.flatten()
    mu, std = w.mean(), w.std()
    x = np.linspace(w.min(), w.max(), 1000)
    plt.hist(w, bins=100, density=True, alpha=0.6)
    plt.plot(x, norm.pdf(x, mu, std))
    plt.title("Embedding vs Normal Distribution")
    plt.savefig(f"figures/{cfg.model.type}/{dataset}/histo.png")
    plt.close()

    w_scaled = w / std  # scale up variance
    stats.probplot(w_scaled, dist="norm", plot=plt)
    plt.title("Q-Q Plot (scaled)")
    plt.savefig(f"figures/{cfg.model.type}/{dataset}/q-q_scaled.png")
    plt.close()
    print(w.mean(), w.std())
    from scipy.stats import skew, kurtosis

    s = skew(w)
    k = kurtosis(w)  # excess kurtosis, 0 for perfect normal
    print(f"Skewness: {s}, Excess kurtosis: {k}")

def main_full_quant_study(path, cfg):
    LARGE_VOCAB, HIDDEN_SIZE = 2_500_000, 256 
    variants = ["FP32", "FP16", "INT8", "LINEAR4", "NF4"]
    results = []
    
    def get_mem(): 
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2

    for name in variants:
        print(f"\n>>> TESTING: {name}")
        dm = DataManager(cfg.model.type, cfg.dataset.name, cfg.training.batch_size, cfg.model.params.max_sequence_length)
        model = Bert4Rec(item_num=dm.num_items, hidden_size=cfg.model.params.hidden_size, 
                         num_layers=cfg.model.params.num_layers, num_heads=cfg.model.params.num_heads,
                         max_sequence_length=cfg.model.params.max_sequence_length)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        if name == "FP32":
            test_for_normal(model, cfg.dataset.name)

        # Add 2.5GB Dummy table (for RAM stress test)
        model.large_test_emb = nn.Embedding(LARGE_VOCAB, HIDDEN_SIZE)
        model.large_test_emb.weight.data.normal_(0, 0.02)

        if name == "FP16":
            model.item_embedding = CPU16bitAbsmaxEmbedding(model.item_embedding)
            model.large_test_emb = CPU16bitAbsmaxEmbedding(model.large_test_emb)
            model.position_embedding = CPU16bitAbsmaxEmbedding(model.position_embedding)
        elif name == "INT8":
            model.item_embedding = CPU8bitAbsmaxEmbedding(model.item_embedding)
            model.large_test_emb = CPU8bitAbsmaxEmbedding(model.large_test_emb)
            model.position_embedding = CPU8bitAbsmaxEmbedding(model.position_embedding)
        elif name == "LINEAR4":
            model.item_embedding = CPU4bitAbsmaxEmbedding(model.item_embedding)
            model.large_test_emb = CPU4bitAbsmaxEmbedding(model.large_test_emb)
            model.position_embedding = CPU4bitAbsmaxEmbedding(model.position_embedding)
        elif name == "NF4":
            model.item_embedding = CPUNF4Embedding(model.item_embedding)
            model.large_test_emb = CPUNF4Embedding(model.large_test_emb)
            model.position_embedding = CPUNF4Embedding(model.position_embedding)
        
        
        gc.collect()
        time.sleep(0.5) 
        
        trainer = RecSysTrainer(model, None, None, "cpu")
        
        monitor = MemoryMonitor()
        monitor_thread = threading.Thread(target=monitor.measure)
        monitor_thread.start()
        
        hr, ndcg, _, _ = trainer.evaluate(dm.test_loader, performance_per_user=True)
        
        monitor.keep_running = False
        monitor_thread.join()
        
        eval_peak_ram = monitor.peak_ram

        total_ram = get_mem()

        # Persistence Check
        p_path = f"tmp_{name}.pth"
        torch.save(model.state_dict(), p_path)
        disk_size = os.path.getsize(p_path) / 1024**2
        os.remove(p_path)
        
        embedding_size = quant_embedding_size_mb(model)
        print(f"[{name}] Item Embedding size: {embedding_size:.1f}MB | Total RAM: {total_ram:.1f}MB | Eval Peak RAM: {eval_peak_ram:.1f}MB | Disk: {disk_size:.1f}MB | NDCG: {ndcg:.4f} | HR: {hr:.4f}")
        
        results.append({
            "variant": name, 
            "item_emb_mb": embedding_size,
            "total_ram_mb": total_ram, 
            "eval_peak_ram_mb": eval_peak_ram, 
            "disk_mb": disk_size, 
            "ndcg": ndcg, 
            "hr": hr
        })
        
        del model, trainer, monitor
        gc.collect()
        time.sleep(1)

    return results

if __name__ == "__main__":
    dataset = "steam" 
    cfg = OmegaConf.load(f"configs/bert/{dataset}.yaml")
    set_seed(cfg.seed)
    
    final_results = main_full_quant_study(f"trained_models/bert_model_{dataset}_{cfg.seed}.pth", cfg)
    
    print("\n" + "="*95 + "\nFINAL QUANTIZATION STUDY RESULTS\n" + "="*95)
    print(pd.DataFrame(final_results).to_string(index=False))
