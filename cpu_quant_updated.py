import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
import pandas as pd
import time
import psutil
import threading
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis, probplot
import argparse

# --- MOCK IMPORTS (Replace these with your actual local imports) ---
# from src.data_manager import DataManager
# from src.models import Bert4Rec
# from src.trainer import RecSysTrainer
# from src.utils import set_seed

# ==========================================
# 1. QUANTIZATION CLASSES
# ==========================================

class CPU16bitAbsmaxEmbedding(nn.Module):
    def __init__(self, source_layer):
        super().__init__()
        self.num_embeddings = source_layer.num_embeddings
        self.embedding_dim = source_layer.embedding_dim
        X = source_layer.weight.data.detach().to(torch.float32)
        self.register_buffer("absmax", X.abs().max() + 1e-8)
        self.register_buffer("c", 65504.0 / self.absmax)
        self.weight_quant = nn.Parameter((self.c * X).half())
        gc.collect()

    def forward(self, x):
        return F.embedding(x, self.weight_quant).float() / self.c

class CPU8bitAbsmaxEmbedding(nn.Module):
    def __init__(self, source_layer):
        super().__init__()
        self.num_embeddings = source_layer.num_embeddings
        self.embedding_dim = source_layer.embedding_dim
        X = source_layer.weight.data.detach().to(torch.float32)
        self.register_buffer("absmax", X.abs().max() + 1e-8)
        self.register_buffer("c", 127.0 / self.absmax)
        self.register_buffer("weight_quant", torch.round(self.c * X).to(torch.int8))
        gc.collect()

    def forward(self, x):
        return F.embedding(x, self.weight_quant).to(torch.float32) / self.c

class CPU4bitAbsmaxEmbedding(nn.Module):
    def __init__(self, source_layer):
        super().__init__()
        self.num_embeddings = source_layer.num_embeddings
        self.embedding_dim = source_layer.embedding_dim
        X = source_layer.weight.data.detach().to(torch.float32)
        self.register_buffer("absmax", X.abs().max() + 1e-8)
        self.register_buffer("c", 7.0 / self.absmax)
        X_uint4 = (torch.round(self.c * X).clamp(-7, 7) + 7).to(torch.uint8)
        self.register_buffer("weight_quant_packed", (X_uint4[:, 0::2] << 4) | X_uint4[:, 1::2])
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
        nf4_lut = [-1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, 
                   -0.09105, 0.0, 0.07958, 0.16093, 0.24611, 0.33791, 0.44071, 0.56261, 0.72295, 1.0]
        self.register_buffer("nf4_lut", torch.tensor(nf4_lut))
        X = source_layer.weight.data.detach().to(torch.float32)
        self.register_buffer("absmax", X.abs().max() + 1e-8)
        self.register_buffer("c", 1.0 / self.absmax)
        X_norm = self.c * X
        indices = torch.zeros((self.num_embeddings, self.embedding_dim), dtype=torch.uint8)
        for i in range(0, self.num_embeddings, 5000):
            end_i = min(i + 5000, self.num_embeddings)
            dist = (X_norm[i:end_i].unsqueeze(-1) - self.nf4_lut).abs()
            indices[i:end_i] = dist.argmin(dim=-1).to(torch.uint8)
        self.register_buffer("weight_quant_packed", (indices[:, 0::2] << 4) | indices[:, 1::2])
        gc.collect()

    def forward(self, x):
        p = F.embedding(x, self.weight_quant_packed)
        idx = torch.stack([(p >> 4) & 0x0F, p & 0x0F], dim=-1).reshape(*x.shape, self.embedding_dim)
        return self.nf4_lut[idx.long()] / self.c

# ==========================================
# 2. UTILITIES & MONITORING
# ==========================================

class MemoryMonitor:
    def __init__(self):
        self.keep_running = True
        self.peak_ram = 0.0
        self.process = psutil.Process(os.getpid())

    def measure(self):
        while self.keep_running:
            try:
                current_ram = self.process.memory_info().rss / (1024 * 1024)
                if current_ram > self.peak_ram: self.peak_ram = current_ram
                time.sleep(0.005)
            except: break

def get_layer_size_mb(layer):
    """ Helper to find weight size regardless of quantization class """
    for name, buf in layer.named_buffers():
        if "weight_quant" in name or "weight" in name:
            return buf.element_size() * buf.numel() / 1024**2
    for name, param in layer.named_parameters():
        if "weight" in name:
            return param.element_size() * param.numel() / 1024**2
    return 0.0

# ==========================================
# 3. CORE LOGIC
# ==========================================

def main_full_quant_study(path, cfg, target_attributes):
    from src.data_manager import DataManager
    from src.models import Bert4Rec, NeuMF
    from src.trainer import RecSysTrainer

    variants = {
        "FP32": None,
        "FP16": CPU16bitAbsmaxEmbedding,
        "INT8": CPU8bitAbsmaxEmbedding,
        "LINEAR4": CPU4bitAbsmaxEmbedding,
        "NF4": CPUNF4Embedding
    }
    
    results = []
    
    for name, q_class in variants.items():
        print(f"\n>>> TESTING: {name}")
        dm = DataManager(cfg.model.type, cfg.dataset.name, cfg, cfg.training.batch_size, cfg.model.params.get("max_sequence_length", 0), smooth_popularity=cfg.training.get("smooth_popularity", False))
        match cfg.model.type:
            case "nmf":
                model = NeuMF(num_users=dm.num_users + 1, num_items=dm.num_items + 1,
                            latent_dim_mf=cfg.model.params.latent_mf, latent_dim_mlp=cfg.model.params.latent_mlp,
                                hidden_sizes=cfg.model.params.hidden_sizes, dropout_prob=cfg.model.params.dropout_rate)
            case "bert":
                model = Bert4Rec(item_num=dm.num_items, hidden_size=cfg.model.params.hidden_size, num_layers=cfg.model.params.num_layers, num_heads=cfg.model.params.num_heads,
                        max_sequence_length=cfg.model.params.max_sequence_length, hidden_dropout=cfg.model.params.hidden_dropout, attention_dropout=cfg.model.params.attention_dropout)
        model.load_state_dict(torch.load(path, map_location="cpu"))

        # Optional: Add Stress-Test Embedding
        model.large_test_emb = nn.Embedding(1_000_000, 128)

        # DYNAMIC REPLACEMENT
        if q_class is not None:
            for attr in target_attributes:
                if hasattr(model, attr):
                    orig_layer = getattr(model, attr)
                    setattr(model, attr, q_class(orig_layer))
        
        gc.collect()
        
        trainer = RecSysTrainer(model, None, None, "cpu")
        monitor = MemoryMonitor()
        m_thread = threading.Thread(target=monitor.measure)
        m_thread.start()
        
        hr, ndcg, _, _ = trainer.evaluate(dm.test_loader, performance_per_user=True)
        
        monitor.keep_running = False
        m_thread.join()

        # Calculate total size of all targeted embeddings
        total_emb_size = sum(get_layer_size_mb(getattr(model, a)) for a in target_attributes if hasattr(model, a))
        
        results.append({
            "variant": name, 
            "target_embs_mb": total_emb_size,
            "eval_peak_ram": monitor.peak_ram, 
            "ndcg": ndcg, 
            "hr": hr
        })
        
        del model, trainer, monitor
        gc.collect()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="ml-1m",
        help="Dataset name (one of 'ml-1m', 'ml-20m', 'beauty', 'steam')"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="nmf",
        help="Model type (one of 'bert', 'nmf')"
    )
    args = parser.parse_args()
    # --- CONFIG ---
    # Update these strings to match your actual file structure
    DATASET = args.dataset
    MODEL_CLASS = args.model
    MODEL_PATH = f"trained_models/{MODEL_CLASS}_model_{DATASET}_42.pth"
    CONFIG_PATH = f"configs/{MODEL_CLASS}/{DATASET}.yaml"
    
    # Define the list of attributes to be replaced
    if MODEL_CLASS == "bert":
        ATTRIBUTES_TO_QUANTIZE = ["item_embedding", "position_embedding", "large_test_emb"]
    else:
        ATTRIBUTES_TO_QUANTIZE = ["embed_user_GMF", "embed_item_GMF", "embed_user_MLP", "embed_item_MLP", "large_test_emb"]

    if os.path.exists(CONFIG_PATH):
        cfg = OmegaConf.load(CONFIG_PATH)
        # Ensure your seed is set for reproducibility
        from src.utils import set_seed
        set_seed(cfg.seed)
        
        final_results = main_full_quant_study(MODEL_PATH, cfg, ATTRIBUTES_TO_QUANTIZE)
        
        print("\n" + "="*80)
        print(pd.DataFrame(final_results).to_string(index=False))
        print("="*80)
    else:
        print(f"Config not found at {CONFIG_PATH}")