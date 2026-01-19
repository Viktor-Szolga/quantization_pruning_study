import torch
import torch.nn as nn
import bitsandbytes as bnb
import gc
from src.models import Bert4Rec
from src.data_manager import MovieLensDataManager
from src.trainer import RecSysTrainer
from pathlib import Path

def get_model_size_vram(model):
    """
    Calculates the memory footprint of the model parameters in MB.
    Handles bitsandbytes quantized parameters correctly.
    """
    total_bits = 0
    for name, param in model.named_parameters():
        # Handle bitsandbytes 4-bit parameters
        if hasattr(param, 'quant_state'):
            # 4 bits per element
            total_bits += param.nelement() * 4
            # Add quantization state overhead (scales, etc.)
            # This is an approximation, but bnb quant_state is usually small
        else:
            # Standard PyTorch parameters
            total_bits += param.nelement() * param.element_size() * 8
            
    for buffer in model.buffers():
        total_bits += buffer.nelement() * buffer.element_size() * 8
        
    return total_bits / (8 * 1024**2) # Convert bits to MB

def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "bert"
    data_manager = MovieLensDataManager(model_type, dataset="ml-1m")
    
    precisions = ["fp32", "fp16", "int8", "nf4"]
    results = {}

    for precision in precisions:
        print(f"\n--- Testing Precision: {precision.upper()} ---")
        
        # 1. Clear GPU memory completely
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated(device) / 1024**2

        # 2. Initialize fresh model
        model = Bert4Rec(
            item_num=data_manager.num_items, 
            hidden_size=256, 
            num_layers=2, 
            num_heads=8,
            max_sequence_length=data_manager.train_set.max_len, 
            dropout=0.2
        ).to(device)
        
        model.load_state_dict(torch.load(Path("trained_models") / Path("best_bert_model_num_steps.pth")))

        # 3. Apply Quantization
        model.set_precision(precision)
        
        # 4. Measure
        model_size_calc = get_model_size_vram(model)
        actual_vram = (torch.cuda.memory_allocated(device) / 1024**2) - initial_mem
        
        # 5. Evaluate
        trainer = RecSysTrainer(model, None, None, device)
        with torch.amp.autocast("cuda"):
            hr, ndcg = trainer.evaluate(data_manager.valid_loader)
        
        print(f"Calculated Model Size: {model_size_calc:.2f} MB")
        print(f"Actual VRAM Increase: {actual_vram:.2f} MB")
        print(f"NDCG: {ndcg:.4f} | HR: {hr:.4f}")
        
        # Check Weight Class
        emb_weight = model.item_embedding.weight
        print(f"Weight Class: {type(emb_weight)}")

    return results

if __name__ == "__main__":
    run_experiment()