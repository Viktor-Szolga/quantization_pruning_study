import torch
import torch.nn as nn
import copy
import time
from bitsandbytes.nn import Embedding4bit, Embedding8bit, Linear4bit
import copy
import torch.nn as nn
import os
import torch.nn.utils.prune as prune
import gc
import pandas as pd
import warnings
import pickle
warnings.filterwarnings(
    "ignore",
    message="Embedding size .* is not divisible by block size"
)
import bitsandbytes as bnb

from src.data_manager import DataManager
from src.models import Bert4Rec
from src.trainer import RecSysTrainer
import torch.nn.functional as F
from src.utils import set_seed
from omegaconf import OmegaConf

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
        input = next(iter(dm.test_loader))[0].to(device)
        _ = model(input)
    print(f"Memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
    print(f"Peak memory used: {torch.cuda.max_memory_allocated(device)/1024**2:.2f} MB")

    return torch.cuda.memory_allocated(device)/1024**2, torch.cuda.max_memory_allocated(device)/1024**2

def evaluate_model(model, dm, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Number of neurons in embedding: {model.item_embedding.weight.numel()}")

    model.to(device)
    for param in model.parameters():
        if isinstance(param, bnb.nn.Params4bit):
            param.data = param.data.to(device)
            # This specific call triggers the internal 'quantized' state
            if hasattr(param, "quant_state"):
                param.quant_state.to(device)


    trainer = RecSysTrainer(model, None, None, device)
    with torch.amp.autocast("cuda"):
        hr, ndcg, user_hr, user_ndcg = trainer.evaluate(dm.test_loader, performance_per_user=True)
         
    print(model.item_embedding.weight.dtype)
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


def get_model_variants(dm, quantized_list, named_parameter, path):
    variants = []
    for value in quantized_list:
        m = Bert4Rec(item_num=dm.num_items, hidden_size=cfg.model.params.hidden_size, num_layers=cfg.model.params.num_layers, num_heads=cfg.model.params.num_heads,
                max_sequence_length=cfg.model.params.max_sequence_length, dropout=cfg.model.params.dropout
            )
        m.load_state_dict(torch.load(path))
        setattr(m, named_parameter, value)
        variants.append(m)
    return variants

def main(path, cfg):
    result_dicts = []
    prune_model = [False, True]
    for p in prune_model:
        dm = DataManager(cfg.model.type, cfg.dataset.name, cfg.training.batch_size, cfg.model.params.max_sequence_length)
        model = Bert4Rec(item_num=dm.num_items, hidden_size=cfg.model.params.hidden_size, num_layers=cfg.model.params.num_layers, num_heads=cfg.model.params.num_heads,
                max_sequence_length=cfg.model.params.max_sequence_length, dropout=cfg.model.params.dropout
            )
        model.load_state_dict(torch.load(path))
        trained_embedding = model.item_embedding
        if p:
            trained_embedding = prune_embedding(trained_embedding)


        test_embedding = nn.Embedding(100000, 10000)
        test_embeddings = get_quantized_embeddings(test_embedding)
        models = get_model_variants(dm, get_quantized_embeddings(trained_embedding), "item_embedding", path)

        for m, test_embedding in zip(models, test_embeddings):
            m.test_embedding = test_embedding
            result_dict = evaluate_model(m, dm)
            result_dicts.append(result_dict)
            m.to("cpu")

            gc.collect()
            torch.cuda.empty_cache()
    return result_dicts

if __name__ == "__main__":
    config_path = "configs/bert/ml-1m.yaml"
    cfg = OmegaConf.load(config_path)
    set_seed(cfg.seed)
    device = "cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else cfg.device


    name = "trained_models/bert_model_ml-1m_42.pth"

    results_list = main(name, cfg)

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
        