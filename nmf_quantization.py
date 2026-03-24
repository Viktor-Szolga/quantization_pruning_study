import torch
import torch.nn as nn
import copy
import time
from bitsandbytes.nn import Embedding4bit, Embedding8bit
import copy
import torch.nn as nn
import os
import torch.nn.utils.prune as prune
import gc
import warnings
import pandas as pd
import pickle
warnings.filterwarnings(
    "ignore",
    message="Embedding size .* is not divisible by block size"
)
import bitsandbytes as bnb

from src.data_manager import DataManager
from src.models import NeuMF
from src.trainer import RecSysTrainer
import torch.nn.functional as F


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
        users, items, labels = next(iter(dm.test_loader))
        if items.dim() == 1:
            items = items.unsqueeze(1)
        batch_size, num_candidate_items = items.shape
        users_flat = users.repeat_interleave(num_candidate_items).to(device)
        items_flat = items.reshape(-1).to(device)

        with torch.amp.autocast("cuda"):
            scores = model(users_flat, items_flat)
    print(f"Memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
    print(f"Peak memory used: {torch.cuda.max_memory_allocated(device)/1024**2:.2f} MB")
    return torch.cuda.memory_allocated(device)/1024**2, torch.cuda.max_memory_allocated(device)/1024**2

def evaluate_model(model, dm, device="cuda" if torch.cuda.is_available() else "cpu"):
    trainer = RecSysTrainer(model, None, None, device)
    with torch.amp.autocast("cuda"):
        hr, ndcg, user_hr, user_ndcg = trainer.evaluate(dm.test_loader, performance_per_user=True)
        
    print(model.item_embedding_mf.weight.dtype)
    print(f"NDCG: {ndcg:.4f} | HR: {hr:.4f}")
    allocated, peak_allocated = print_size(model, dm)
    print(f"---"*10)
    return {
        "ndcg": ndcg,
        "hr": hr,
        "user_hr": user_hr,
        "user_ndcg": user_ndcg,
        "allocated": allocated,
        "peak_allocated": peak_allocated
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



def main(path):
    result_dicts = []
    prune_model = [False, True]
    for p in prune_model:
        
        data_manager = DataManager(cfg.model.type, cfg.dataset.name, cfg, cfg.training.batch_size, cfg.model.params.get("max_sequence_length", 0), smooth_popularity=cfg.training.get("smooth_popularity", False))
        model = NeuMF(num_users=data_manager.num_users + 1, num_items=data_manager.num_items + 1,
                           latent_dim_mf=cfg.model.params.latent_mf, latent_dim_mlp=cfg.model.params.latent_mlp,
                             hidden_sizes=cfg.model.params.hidden_sizes, dropout_prob=cfg.model.params.dropout_rate)
        
        model.load_state_dict(torch.load(path, map_location="cuda"))
        
        embeddings = [model.item_embedding_mf, model.item_embedding_mlp, model.user_embedding_mf, model.user_embedding_mlp]
        pruned_embeddings=[]
        if p:
            for embed in embeddings:
                pruned_embeddings.append(prune_embedding(embed))
            embeddings = pruned_embeddings

        test_embedding = nn.Embedding(100000, 10000)
        test_embeddings = get_quantized_embeddings(test_embedding)
        quantized_embeddings = []
        for embed in embeddings:
            quantized_embeddings.append(get_quantized_embeddings(embed))

        models = get_model_variants(model, quantized_embeddings, ["item_embedding_mf", "item_embedding_mlp", "user_embedding_mf", "user_embedding_mlp"])

        for m, test_embedding in zip(models, test_embeddings):
            m.test_embedding = test_embedding
            result_dict = evaluate_model(m, dm)
            result_dicts.append(result_dict)
            m.to("cpu")

            gc.collect()
            torch.cuda.empty_cache()
    return result_dicts

            

if __name__ == "__main__":
    name = "trained_models/nmf_model_ml-1m_42.pth"
    results_list = main(name)

    type_classifications = ["fp32_np","fp16_np", "int8_np", "fp4_np", "nf4_np", "fp32_p","fp16_p", "int8_p", "fp4_p", "nf4_p"]

    for result, type_classification in zip(results_list, type_classifications):
        user_df = pd.DataFrame({
            "user_id": list(result["user_hr"].keys()),
            "hr": list(result["user_hr"].values()),
            "ndcg": [result["user_ndcg"][u] for u in result["user_hr"].keys()]
        })

        os.makedirs(f"per_user_results", exist_ok=True)
        user_df.to_csv(f"per_user_results/{name[15:-4]}_{type_classification}.csv", index=False)
    with open("results.pkl", "wb") as f:
        pickle.dump(result, f)