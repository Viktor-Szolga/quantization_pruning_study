import torch
import torch.nn as nn
import copy
import time
from bitsandbytes.nn import Embedding4bit, Embedding8bit
from codecarbon import EmissionsTracker
import copy
import torch.nn as nn
import pandas as pd
import os
import torch.nn.utils.prune as prune
import gc
import warnings
warnings.filterwarnings(
    "ignore",
    message="Embedding size .* is not divisible by block size"
)
    
# Note: You may need to run: pip install bitsandbytes
try:
    import bitsandbytes as bnb
except ImportError:
    print("Please install bitsandbytes: pip install bitsandbytes")

from src.data_manager import MovieLensDataManager
from src.models import NeuralMF
from src.trainer import RecSysTrainer
import torch.nn.functional as F


def check_quantization_norm(module_8_bit, module_4_bit):
    # placeholder/reminder
    import torch

    diff_8bit = torch.norm(trained_embedding.weight - module_8_bit.weight.float())
    diff_4bit = torch.norm(trained_embedding.weight - module_4_bit.weight.float())
    print("8-bit embedding diff:", diff_8bit.item())
    print("4-bit embedding diff:", diff_4bit.item())

def prune_embedding(embedding, amount=0.1, unstructured=True):
    if unstructured:
        module = copy.deepcopy(embedding)
        prune.random_unstructured(module, name="weight", amount=amount)
        prune.remove(module, 'weight')
        return module
    else:
        pass


def print_size(model, dm, device="cuda" if torch.cuda.is_available() else "cpu"):
    device = "cuda"
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    with torch.no_grad():
        users, items, labels = next(iter(dm.valid_loader))
        if items.dim() == 1:
            items = items.unsqueeze(1)
        batch_size, num_candidate_items = items.shape
        users_flat = users.repeat_interleave(num_candidate_items).to(device)
        items_flat = items.reshape(-1).to(device)

        with torch.amp.autocast("cuda"):
            scores = model(users_flat, items_flat)
    memory_allocated = torch.cuda.memory_allocated(device)/1024**2
    memory_peak = torch.cuda.max_memory_allocated(device)/1024**2
    print(f"Memory allocated: {memory_allocated:.2f} MB")
    print(f"Peak memory used: {memory_peak:.2f} MB")
    return memory_allocated, memory_peak


def evaluate_model(model, dm, experiment_name, device="cuda" if torch.cuda.is_available() else "cpu"):
    tracker = EmissionsTracker(
        project_name="nmf_emissions",
        experiment_id=experiment_name,
        output_dir="emissions/nmf",
        log_level="error"
    )
    trainer = RecSysTrainer(model, None, None, device)
    tracker.start()
    with torch.amp.autocast("cuda"):
        hr1, ndcg1 = trainer.evaluate(dm.test_loader, k=1)
        hr5, ndcg5 = trainer.evaluate(dm.test_loader, k=5)
        hr10, ndcg10 = trainer.evaluate(dm.test_loader, k=10)
    emissions = tracker.stop()
    print(model.item_embedding_mf.weight.dtype)
    print(f"NDCG: {ndcg1:.4f} | HR: {hr1:.4f}")
    allocated, peak = print_size(model, dm)
    print(f"{experiment_name}: {emissions:.6f} kgCO₂eq")
    print(f"---"*10)
    return {
        "hr1": hr1,
        "ndcg1": ndcg1,
        "hr5": hr5,
        "ndcg5": ndcg5,
        "hr10": hr10,
        "ndcg10": ndcg10,
        "emissions": emissions,
        "allocated": allocated,
        "peak": peak
    }  

def get_quantized_embeddings(embedding, device = "cuda" if torch.cuda.is_available() else "cpu"):
    quantized = []
    embedding_half = copy.deepcopy(embedding).half()
    module_4_bit = Embedding4bit(embedding.weight.shape[0], embedding.weight.shape[1], quant_type="fp4")
    module_4_bit.load_state_dict(embedding.state_dict())
    
    module_4_bit_nf = Embedding4bit(embedding.weight.shape[0], embedding.weight.shape[1], quant_type="nf4")
    module_4_bit_nf.load_state_dict(embedding.state_dict())

    module_8_bit = Embedding8bit(embedding.weight.shape[0], embedding.weight.shape[1])
    module_8_bit.load_state_dict(embedding.state_dict())

    quantized = [embedding, embedding_half, module_8_bit, module_4_bit, module_4_bit_nf]
    return quantized


def get_model_variants(model, quantized_list, named_parameter):
    variants = []
    for i in range(len(quantized_list[0])):
        m = copy.deepcopy(model)
        for j, parameter in enumerate(named_parameter):       
            setattr(m, parameter, quantized_list[j][i])
        variants.append(m)
    return variants


if __name__ == "__main__":
    path = "emissions/nmf/emissions.csv"

    if os.path.exists(path):
        os.remove(path)
    prune_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    data_dict = {}
    for p_level in prune_levels:
        
        dm = MovieLensDataManager("nmf")
        model = NeuralMF(num_users=dm.num_users+1, num_items=dm.num_items+1, latent_mf=32, latent_mlp=512, hidden_sizes=[512, 256, 128, 64])
        
        if os.path.exists("trained_models/best_nmf_model.pth"):
            model.load_state_dict(torch.load("trained_models/best_nmf_model.pth", map_location="cuda"))
        
        embeddings = [model.item_embedding_mf, model.item_embedding_mlp, model.user_embedding_mf, model.user_embedding_mlp]

        pruned_embeddings=[]
        if p_level:
            for embed in embeddings:
                pruned_embeddings.append(prune_embedding(embed, amount=p_level))
            embeddings = pruned_embeddings

        test_embedding = nn.Embedding(100000, 10000)
        test_embeddings = get_quantized_embeddings(test_embedding)
        quantized_embeddings = []
        for embed in embeddings:
            quantized_embeddings.append(get_quantized_embeddings(embed))

        models = get_model_variants(model, quantized_embeddings, ["item_embedding_mf", "item_embedding_mlp", "user_embedding_mf", "user_embedding_mlp"])
        names = ["fp32", "fp16", "int8", "fp4", "nf4"]
        if p_level:
            names = [name + f"_pruned_{p_level}" for name in names]
        for m, test_embedding, name in zip(models, test_embeddings, names):
            #m.test_embedding = test_embedding
            data_dict[name] = evaluate_model(m, dm, name)
            m.to("cpu")

            gc.collect()
            torch.cuda.empty_cache()
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df.index.name = "run_name"

    df.to_csv("results/nmf.csv")