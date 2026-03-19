"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import gc
import os
import pandas as pd
import warnings
import bitsandbytes as bnb
from bitsandbytes.nn import Embedding4bit, Embedding8bit
from omegaconf import OmegaConf

# Suppress the specific embedding block size warning
warnings.filterwarnings("ignore", message="Embedding size .* is not divisible by block size")

def warmup_gpu():
    '''Initializes CUDA and bitsandbytes kernels so overhead is consistent.'''
    print("Warming up GPU and loading kernels...")
    dummy = torch.randn(100, 100).cuda()
    # Loading a small 8bit layer forces bitsandbytes to initialize its C-libraries
    _ = Embedding8bit(10, 10).cuda()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def get_memory_stats(device="cuda"):
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    return allocated, peak

def evaluate_mock(model, device="cuda"):
    '''Simulates a forward pass to measure peak VRAM correctly.'''
    model.to(device)
    for param in model.parameters():
        if hasattr(param, "quant_state"):
            param.quant_state.to(device)

    # Static footprint
    static_mem, _ = get_memory_stats(device)
    
    # Mock Forward Pass
    with torch.no_grad():
        dummy_input = torch.randint(0, 100, (1, 50)).to(device)
        _ = model(dummy_input)
    
    _, peak_mem = get_memory_stats(device)
    
    dtype = model.item_embedding.weight.dtype
    print(f"-> Dtype: {dtype} | Static: {static_mem:.2f}MB | Peak: {peak_mem:.2f}MB")
    
    return static_mem, peak_mem

def run_experiment(item_count, hidden_size):
    print(f"\n{'='*20}")
    print(f"RUNNING SCALE: {item_count} items, {hidden_size} hidden size")
    print(f"{'='*20}")
    
    # 1. Create Base Embedding (FP32)
    # We use a standard NN module first
    base_emb = nn.Embedding(item_count, hidden_size)
    
    # 2. Define Variants
    # Standard, 8-bit, and 4-bit (NF4)
    variants = [
        ("FP32", lambda: copy.deepcopy(base_emb).float()),
        ("FP16", lambda: copy.deepcopy(base_emb).half()),
        ("INT8", lambda: Embedding8bit(item_count, hidden_size)),
        ("NF4 ", lambda: Embedding4bit(item_count, hidden_size, quant_type="nf4"))
    ]

    for name, loader in variants:
        # Create a dummy container model for the embedding
        class Wrapper(nn.Module):
            def __init__(self, emb):
                super().__init__()
                self.item_embedding = emb
            def forward(self, x):
                return self.item_embedding(x)

        model_layer = loader()
        if hasattr(model_layer, "load_state_dict") and not isinstance(model_layer, nn.Embedding):
             model_layer.load_state_dict(base_emb.state_dict())
        
        m = Wrapper(model_layer)
        print(f"Testing {name}: ", end="")
        evaluate_mock(m)
        
        del m, model_layer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    warmup_gpu()
    
    # SCENARIO A: Your current model size (Approx 1M neurons)
    # Weights are ~4MB. Metadata overhead will dominate here.
    run_experiment(item_count=8000, hidden_size=128)

    # SCENARIO B: Large model size (Approx 64M neurons)
    # Weights are ~256MB. Here, 4-bit SHOULD show massive savings.
    run_experiment(item_count=100000, hidden_size=640)
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import gc
import os
import pandas as pd
import warnings
import bitsandbytes as bnb
from bitsandbytes.nn import Embedding4bit, Embedding8bit
from omegaconf import OmegaConf

# Internal project imports (adjust if your structure differs)
from src.data_manager import DataManager
from src.models import Bert4Rec
from src.trainer import RecSysTrainer
from src.utils import set_seed

warnings.filterwarnings("ignore", message="Embedding size .* is not divisible by block size")

def get_model_size_breakdown(model):
    """
    Calculates the exact byte-size of a model by inspecting every parameter.
    This bypasses CUDA's library overhead and looks at the actual TENSORS.
    """
    total_bytes = 0
    for name, param in model.named_parameters():
        if hasattr(param, "quant_state"):
            # The weights (packed in uint8)
            p_bytes = param.data.nelement() * param.data.element_size()
            
            # The Quantization State (Scales, Offsets, Codebooks)
            q_state = param.quant_state
            qs_bytes = 0
            
            # Sum up tensors in quant_state
            if hasattr(q_state, 'absmax') and torch.is_tensor(q_state.absmax):
                qs_bytes += q_state.absmax.nelement() * q_state.absmax.element_size()
            if hasattr(q_state, 'state2') and q_state.state2: # Double quant
                if hasattr(q_state.state2, 'absmax') and torch.is_tensor(q_state.state2.absmax):
                    qs_bytes += q_state.state2.absmax.nelement() * q_state.state2.absmax.element_size()
            
            layer_total = p_bytes + qs_bytes
            total_bytes += layer_total
        else:
            p_bytes = param.nelement() * param.element_size()
            total_bytes += p_bytes
            
    return total_bytes / 1024**2

def prune_embedding(embedding, amount=0.1):
    module = copy.deepcopy(embedding)
    prune.random_unstructured(module, name="weight", amount=amount)
    prune.remove(module, 'weight')
    return module

def evaluate_variant(variant_name, embedding_layer, base_model, dm, device="cuda"):
    """Evaluates a model variant and returns performance + true memory size."""
    model = copy.deepcopy(base_model)
    model.item_embedding = embedding_layer
    model.to(device)
    
    # Ensure quantization metadata is on GPU
    for param in model.parameters():
        if hasattr(param, "quant_state"):
            param.quant_state.to(device)

    # 1. CALCULATE TRUE MODEL SIZE (The 'Guest' size)
    true_size_mb = get_model_size_breakdown(model)
    
    # 2. EVALUATE PERFORMANCE
    trainer = RecSysTrainer(model, None, None, device)
    with torch.amp.autocast("cuda"):
        hr, ndcg, user_hr, user_ndcg = trainer.evaluate(dm.test_loader, performance_per_user=True)
    
    # 3. MEASURE PEAK VRAM (The 'Worktable' size)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2

    print(f"[{variant_name:10}] NDCG: {ndcg:.4f} | Net Model Size: {true_size_mb:.2f} MB | Peak VRAM: {peak_mem:.2f} MB")
    
    # Clean up
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    return {
        "tag": variant_name, "ndcg": ndcg, "hr": hr, 
        "user_hr": user_hr, "user_ndcg": user_ndcg, 
        "net_size": true_size_mb, "peak": peak_mem
    }

def main(path, cfg):
    set_seed(cfg.seed)
    dm = DataManager(cfg.model.type, cfg.dataset.name, cfg.training.batch_size, cfg.model.params.max_sequence_length)
    
    # Load Skeleton
    base_model = Bert4Rec(
        item_num=dm.num_items, hidden_size=cfg.model.params.hidden_size, 
        num_layers=cfg.model.params.num_layers, num_heads=cfg.model.params.num_heads,
        max_sequence_length=cfg.model.params.max_sequence_length, dropout=cfg.model.params.dropout
    )
    base_model.load_state_dict(torch.load(path))
    
    results = []
    for is_pruned in [False, True]:
        p_tag = "p" if is_pruned else "np"
        print(f"\n--- Testing {'Pruned' if is_pruned else 'Non-Pruned'} Variants ---")
        
        source_emb = base_model.item_embedding
        if is_pruned:
            source_emb = prune_embedding(source_emb)
        
        row, col = source_emb.weight.shape
        
        # Define variants
        configs = [
            ("fp32", lambda: nn.Embedding(row, col)),
            ("fp16", lambda: nn.Embedding(row, col).half()),
            ("int8", lambda: Embedding8bit(row, col)),
            ("nf4",  lambda: Embedding4bit(row, col, quant_type="nf4"))
        ]
        
        for name, layer_factory in configs:
            layer = layer_factory()
            if hasattr(layer, 'load_state_dict'):
                layer.load_state_dict(source_emb.state_dict())
            
            variant_label = f"{name}_{p_tag}"
            res = evaluate_variant(variant_label, layer, base_model, dm)
            results.append(res)
            
    return results

if __name__ == "__main__":
    CONFIG_PATH = "configs/bert/ml-1m.yaml"
    MODEL_PATH = "trained_models/bert_model_ml-1m_42.pth"
    
    if os.path.exists(MODEL_PATH):
        config = OmegaConf.load(CONFIG_PATH)
        final_results = main(MODEL_PATH, config)

        os.makedirs("per_user_results", exist_ok=True)
        for r in final_results:
            df = pd.DataFrame({
                "u_id": list(r["user_hr"].keys()),
                "hr": list(r["user_hr"].values()),
                "ndcg": [r["user_ndcg"][k] for k in r["user_hr"].keys()]
            })
            df.to_csv(f"per_user_results/{r['tag']}.csv", index=False)
        print("\nExperiment finished. Net Model Size reflects actual weights + metadata.")
    else:
        print(f"Model path {MODEL_PATH} not found.")



