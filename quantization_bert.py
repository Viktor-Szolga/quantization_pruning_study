import torch
import torch.nn as nn
import copy
import time
import os
# Note: You may need to run: pip install bitsandbytes
try:
    import bitsandbytes as bnb
except ImportError:
    print("Please install bitsandbytes: pip install bitsandbytes")

from src.data_manager import MovieLensDataManager
from src.models import Bert4Rec
from src.trainer import RecSysTrainer

def measure_gpu_metrics(model, data_loader, description, compute_dtype=torch.float32):
    print(f"🚀 Testing {description} on GPU...")
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    
    # 1. Accuracy
    trainer = RecSysTrainer(model, None, None, device)
    hr, ndcg = trainer.evaluate(data_loader)
    
    # 2. Latency (with CUDA Synchronization for accuracy)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs = batch[0].to(device)
            
            starter.record()
            # Use autocast for FP16 to trigger Tensor Cores
            with torch.cuda.amp.autocast(enabled=(compute_dtype == torch.float16)):
                _ = model(inputs)
            ender.record()
            
            torch.cuda.synchronize()
            if i > 10:  # Warmup
                timings.append(starter.elapsed_time(ender))
            if i >= 60: break 
            
    avg_latency = sum(timings) / len(timings) / 1000 # Convert ms to s

    # 3. VRAM Usage (Approximate)
    vram_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    
    return {
        "Config": description,
        "VRAM (MB)": vram_mb,
        "Latency (s)": avg_latency,
        "NDCG": ndcg,
        "HR": hr
    }

def main():
    if not torch.cuda.is_available():
        print("❌ CUDA not found. This script requires a GPU.")
        return

    dm = MovieLensDataManager("bert")
    # Load base model structure
    base = Bert4Rec(dm.num_items, 128, 8, 4, dm.train_set.max_len)
    base.load_state_dict(torch.load("testing/best_bert_model.pth"))
    
    results = []

    # --- 1. FP32 (Full Precision) ---
    results.append(measure_gpu_metrics(copy.deepcopy(base), dm.valid_loader, "FP32 (Base)"))

    # --- 2. FP16 (Half Precision) ---
    # We simply cast the model to half. This is native on GPU.
    m_fp16 = copy.deepcopy(base).half()
    results.append(measure_gpu_metrics(m_fp16, dm.valid_loader, "FP16 (Half)", compute_dtype=torch.float16))

    # --- 3. INT8 (Using bitsandbytes) ---
    # This replaces Linear layers with 8-bit versions
    print("📦 Loading INT8 via bitsandbytes...")
    m_int8 = copy.deepcopy(base)
    for name, module in m_int8.named_children():
        if isinstance(module, nn.Linear):
            # Replace with bnb 8-bit linear
            new_layer = bnb.nn.Linear8bitLt(
                module.in_features, module.out_features, 
                bias=module.bias is not None, has_fp16_weights=False
            )
            new_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_layer.bias.data = module.bias.data.clone()
            setattr(m_int8, name, new_layer)
    results.append(measure_gpu_metrics(m_int8, dm.valid_loader, "INT8 (bnb)"))

    # --- 4. INT4 (Using bitsandbytes NF4) ---
    print("💎 Loading INT4 (NF4) via bitsandbytes...")
    m_int4 = copy.deepcopy(base)
    for name, module in m_int4.named_children():
        if isinstance(module, nn.Linear):
            new_layer = bnb.nn.Linear4bit(
                module.in_features, module.out_features, 
                bias=module.bias is not None, quant_type="nf4"
            )
            # Weights are quantized during assignment in bnb
            setattr(m_int4, name, new_layer)
    results.append(measure_gpu_metrics(m_int4, dm.valid_loader, "INT4 (NF4)"))

    # --- FINAL TABLE ---
    print("\n" + "="*95)
    print(f"{'Config':<20} | {'VRAM (MB)':<12} | {'Latency (s)':<12} | {'NDCG':<10} | {'HR':<10}")
    print("-" * 95)
    for r in results:
        print(f"{r['Config']:<20} | {r['VRAM (MB)']:<12.2f} | {r['Latency (s)']:<12.4f} | {r['NDCG']:<10.4f} | {r['HR']:<10.4f}")
    print("="*95)

if __name__ == "__main__":
    main()