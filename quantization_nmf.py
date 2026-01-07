import torch
import torch.nn as nn
import copy
import time
import numpy as np
import os
try:
    import bitsandbytes as bnb
except ImportError:
    print("Please install bitsandbytes: pip install bitsandbytes")

from src.data_manager import MovieLensDataManager
from src.models import NeuralMF
from src.trainer import RecSysTrainer

def get_disk_size(model):
    """Saves model to a temp file to measure actual storage size."""
    temp_path = "temp_size_check.pth"
    torch.save(model.state_dict(), temp_path)
    size_bytes = os.path.getsize(temp_path)
    os.remove(temp_path)
    return size_bytes / 1024  # Return in KB

def replace_with_quantized(model, method="int8"):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if method == "int8":
                new_layer = bnb.nn.Linear8bitLt(
                    module.in_features, module.out_features, 
                    bias=module.bias is not None, has_fp16_weights=False
                )
            else: # int4
                new_layer = bnb.nn.Linear4bit(
                    module.in_features, module.out_features, 
                    bias=module.bias is not None, quant_type="nf4"
                )
            if method == "int8":
                new_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_layer.bias.data = module.bias.data.clone()
            setattr(model, name, new_layer)
        else:
            replace_with_quantized(module, method)
    return model

def measure_gpu_metrics(model, data_loader, description, compute_dtype=torch.float32):
    print(f"🚀 Testing {description} on GPU...")
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    
    # 1. Get Disk Size (Real Compression)
    disk_kb = get_disk_size(model)
    
    # 2. Accuracy
    trainer = RecSysTrainer(model, None, torch.nn.BCEWithLogitsLoss(), device)
    hr, ndcg = trainer.evaluate(data_loader)
    
    # 3. Latency
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            users, items, _ = [b.to(device) for b in batch]
            if items.dim() == 1: items = items.unsqueeze(1)
            batch_size, num_items = items.shape

            starter.record()
            with torch.amp.autocast('cuda', enabled=(compute_dtype == torch.float16)):
                users_flat = users.repeat_interleave(num_items)
                items_flat = items.reshape(-1)
                _ = model(users_flat, items_flat)
            ender.record()
            
            torch.cuda.synchronize()
            if i > 10: timings.append(starter.elapsed_time(ender))
            if i >= 60: break 
            
    avg_latency = (sum(timings) / len(timings)) / 1000 
    vram_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    
    return {
        "Config": description, 
        "VRAM (MB)": vram_mb, 
        "Disk (KB)": disk_kb,
        "Latency (s)": avg_latency, 
        "NDCG": ndcg, 
        "HR": hr
    }

def main():
    dm = MovieLensDataManager("nmf")
    base = NeuralMF(num_users=dm.num_users+1, num_items=dm.num_items+1, latent_mf=4, latent_mlp=32)
    
    if os.path.exists("testing/best_nmf_model.pth"):
        base.load_state_dict(torch.load("testing/best_nmf_model.pth", map_location="cuda"))
    
    results = []
    
    # Baseline FP32
    results.append(measure_gpu_metrics(copy.deepcopy(base), dm.valid_loader, "FP32"))
    
    # FP16
    results.append(measure_gpu_metrics(copy.deepcopy(base).half(), dm.valid_loader, "FP16", compute_dtype=torch.float16))
    
    # INT8
    m_int8 = replace_with_quantized(copy.deepcopy(base), method="int8")
    results.append(measure_gpu_metrics(m_int8, dm.valid_loader, "INT8 (bnb)"))
    
    # INT4
    m_int4 = replace_with_quantized(copy.deepcopy(base), method="int4")
    results.append(measure_gpu_metrics(m_int4, dm.valid_loader, "INT4 (NF4)"))

    print("\n" + "="*110)
    print(f"{'Config':<15} | {'VRAM (MB)':<12} | {'Disk (KB)':<12} | {'Latency (s)':<12} | {'NDCG':<10} | {'HR':<10}")
    print("-" * 110)
    for r in results:
        print(f"{r['Config']:<15} | {r['VRAM (MB)']:<12.2f} | {r['Disk (KB)']:<12.2f} | {r['Latency (s)']:<12.4f} | {r['NDCG']:<10.4f} | {r['HR']:<10.4f}")
    print("="*110)

if __name__ == "__main__":
    main()