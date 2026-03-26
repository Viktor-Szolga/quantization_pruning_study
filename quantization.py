import argparse
import time
import os
import gc
import threading
import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import torch.nn.utils.prune as prune
from src.utils import set_seed
from src.data_manager import DataManager
from src.models import NeuMF, Bert4Rec
from src.trainer import RecSysTrainer
import pickle
from src.utils import CUDAEnergyMonitor, CUDAMemoryMonitor, TiedEmbeddingLinear, BNBFP4Embedding, BNBNF4Embedding, BNB8bitEmbedding, GPU16bitEmbedding


warnings.filterwarnings("ignore", category=UserWarning, message=".*nested tensors.*")

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / 1024**2 # Size in MB
def prune_embedding(layer, sparsity=0.5):
    if sparsity == 0:
        return layer
    prune.l1_unstructured(layer, name="weight", amount=sparsity)
    prune.remove(layer, "weight")
    return layer

def main(config_path, seed, precision, device, sparsity):
    cfg = OmegaConf.load(config_path)
    cfg.training.batch_size=64
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("This script requires a CUDA-enabled GPU.")

    data_manager = DataManager(cfg.model.type, cfg.dataset.name, cfg, cfg.training.batch_size, cfg.model.params.get("max_sequence_length", 0), smooth_popularity=cfg.training.get("smooth_popularity", False))
    monitor = CUDAMemoryMonitor(device)
    energy_monitor = CUDAEnergyMonitor(device_index=0)

    
    variants = {
        "fp32": None,
        "fp16": GPU16bitEmbedding,
        "int8": BNB8bitEmbedding,
        "fp4": BNBFP4Embedding,
        "nf4": BNBNF4Embedding
    }    

    if cfg.model.type == "bert":
        target_attributes = ["item_embedding", "position_embedding", "large_test_emb"]
    else:
        target_attributes = ["embed_user_GMF", "embed_item_GMF", "embed_user_MLP", "embed_item_MLP", "large_test_emb"]
    
    results = []
    raw_results = []
    temp_model_file = "trained_models/temp_quantized_model.pth"

    for name in precision:
        print(f"\n>>> TESTING: {name}")
        eval_mean_ram = []
        eval_peak_ram = []
        energy_usage = []
        ellapsed_time = []
        
        for _ in range(2):
            if 'model' in locals(): del model
            if 'trainer' in locals(): del trainer
            if 'orig_layer' in locals(): del orig_layer
            gc.collect()
            torch.cuda.empty_cache()
            
            q_class = variants[name]
            set_seed(seed)
            
            match cfg.model.type:
                case "nmf":
                    model = NeuMF(num_users=data_manager.num_users + 1, num_items=data_manager.num_items + 1,
                                latent_dim_mf=cfg.model.params.latent_mf, latent_dim_mlp=cfg.model.params.latent_mlp,
                                    hidden_sizes=cfg.model.params.hidden_sizes, dropout_prob=cfg.model.params.dropout_rate)
                case "bert":
                    model = Bert4Rec(item_num=data_manager.num_items, hidden_size=cfg.model.params.hidden_size, num_layers=cfg.model.params.num_layers, num_heads=cfg.model.params.num_heads,
                            max_sequence_length=cfg.model.params.max_sequence_length, hidden_dropout=cfg.model.params.hidden_dropout, attention_dropout=cfg.model.params.attention_dropout)
        
            model.load_state_dict(torch.load(f"{cfg.saving.save_dir}/{cfg.saving.filename}_{cfg.dataset.name}_{seed}.pth", map_location="cpu"))
            model.large_test_emb = torch.nn.Embedding(1_000_000, 128)
            model = model.to(device)
            if sparsity > 0.0:
                for attr in target_attributes:
                    if hasattr(model, attr):
                        orig_layer = getattr(model, attr)
                        setattr(model, attr, prune_embedding(orig_layer, sparsity))
                        del orig_layer
            
            if q_class is not None:
                for attr in target_attributes:
                    if hasattr(model, attr):
                        orig_layer = getattr(model, attr)
                        setattr(model, attr, q_class(orig_layer))
                        del orig_layer
            if cfg.model.type == "bert":
                model.output_layer = TiedEmbeddingLinear(model.item_embedding).to(device)
            torch.save(model, temp_model_file)
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
            

            monitor.calibrate()
            model = torch.load(temp_model_file, map_location=device, weights_only=False)
            
            trainer = RecSysTrainer(model, None, None, device)
            
            energy_monitor.calibrate()

            e_thread = threading.Thread(target=energy_monitor.measure)
            m_thread = threading.Thread(target=monitor.measure)
            e_thread.start()
            m_thread.start()
            t_start = time.perf_counter()
            trainer.measure_metrics(data_manager.test_loader)
            t_end = time.perf_counter()
            monitor.keep_running = False
            energy_monitor.keep_running = False
            m_thread.join()
            e_thread.join()
            
            eval_mean_ram.append(monitor.mean_vram)
            eval_peak_ram.append(monitor.peak_vram)
            energy_usage.append(energy_monitor.total_energy_mj / 1000)
            ellapsed_time.append(t_end - t_start)
            
            if os.path.exists(temp_model_file):
                os.remove(temp_model_file)
        
        hr, ndcg, hr_user, ndcg_user = trainer.evaluate(data_manager.test_loader, performance_per_user=True) 
        results.append({
            "variant": name, 
            "sparsity": sparsity,
            "model_size": get_model_size(model),
            "mean_ram_mean": np.array(eval_mean_ram[1:]).mean(), 
            "peak_ram_mean": np.array(eval_peak_ram[1:]).mean(), 
            "mean_ram_std": np.array(eval_mean_ram[1:]).std(), 
            "energy_j": np.array(energy_usage[1:]).mean(),
            "time": np.array(ellapsed_time[1:]).mean(),
            "ndcg": ndcg, 
            "hr": hr
        })
        raw_results.append({
            "variant": name, 
            "sparsity": sparsity,
            "model_size": get_model_size(model),
            "mean_ram_mean": np.array(eval_mean_ram[1:]), 
            "peak_ram_mean": np.array(eval_peak_ram[1:]), 
            "mean_ram_std": np.array(eval_mean_ram[1:]), 
            "energy_j": np.array(energy_usage[1:]),
            "time": np.array(ellapsed_time[1:]),
            "ndcg_per_user": hr_user, 
            "hr_per_user": ndcg_user,
        })
    return results, raw_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="bert", help="Model type (one of 'bert', 'nmf')")
    parser.add_argument("-d", "--dataset", type=str, default="ml-1m", help="Dataset (one of 'ml-1m', 'ml-20m', 'beauty', 'steam')")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Seed to to be used (requires model with that seed)")
    parser.add_argument("-p", "--precisions", nargs="+", type=str, default=["fp32"], help="Nargs with (fp32, fp16, int8, fp4, nf4)")
    parser.add_argument("--sparsity", type=float, default=0.0, help="Pruning sparsity to test e.g. 0.0 0.3 0.5 0.7")
    
    args = parser.parse_args()
    results, raw_results = main(f"configs/{args.model}/{args.dataset}.yaml", args.seed, args.precisions, "cuda", args.sparsity)
    results = pd.DataFrame(results)
    print("\n" + "="*90)
    print(results.to_string(index=False))
    print("="*90)
    os.makedirs(f"study_results/{args.model}", exist_ok=True)
    results.to_csv(f"study_results/{args.model}/{args.dataset}_{args.seed}_{args.sparsity}.csv")
    with open(f"study_results/{args.model}/{args.dataset}_{args.seed}_{args.sparsity}.pkl", "wb") as f:
        pickle.dump(raw_results, f)