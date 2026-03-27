import torch
import torch.nn as nn
import copy
import gc
import os
import pickle
import warnings
import pandas as pd

import bitsandbytes as bnb
from bitsandbytes.nn import Embedding8bit

from torchao.quantization import quantize_, int4_weight_only, int8_weight_only

warnings.filterwarnings(
    "ignore",
    message="Embedding size .* is not divisible by block size"
)

import torch.nn.utils.prune as prune
from omegaconf import OmegaConf

from src.data_manager import DataManager
from src.models import Bert4Rec
from src.trainer import RecSysTrainer
from src.utils import set_seed


# ── Pruning ───────────────────────────────────────────────────────────────────

def prune_embedding(embedding, amount=0.1):
    module = copy.deepcopy(embedding)
    prune.random_unstructured(module, name="weight", amount=amount)
    prune.remove(module, "weight")
    return module


# ── Memory reporting ──────────────────────────────────────────────────────────

def print_size(model, dm, device="cuda"):
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    with torch.no_grad():
        inp = next(iter(dm.test_loader))[0].to(device)
        _ = model(inp)
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"Memory allocated: {allocated:.2f} MB")
    print(f"Peak memory used: {peak:.2f} MB")
    return allocated, peak


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, dm, device="cuda"):
    print(f"Number of neurons in embedding: {model.item_embedding.weight.numel()}")
    model.to(device)
    trainer = RecSysTrainer(model, None, None, device)
    with torch.amp.autocast("cuda"):
        hr, ndcg, user_hr, user_ndcg = trainer.evaluate(dm.test_loader, performance_per_user=True)
    print(f"Embedding dtype: {model.item_embedding.weight.dtype}")
    print(f"NDCG: {ndcg:.4f} | HR: {hr:.4f}")
    allocated, peak = print_size(model, dm, device)
    print("---" * 10)
    return {"ndcg": ndcg, "hr": hr, "user_hr": user_hr, "user_ndcg": user_ndcg,
            "allocated": allocated, "peak_allocated": peak}


# ── Model loading helpers ─────────────────────────────────────────────────────

def load_fresh_model(dm, cfg, path):
    model = Bert4Rec(
        item_num=dm.num_items,
        hidden_size=cfg.model.params.hidden_size,
        num_layers=cfg.model.params.num_layers,
        num_heads=cfg.model.params.num_heads,
        max_sequence_length=cfg.model.params.max_sequence_length,
        dropout=cfg.model.params.dropout,
    )
    model.load_state_dict(torch.load(path))
    return model


def apply_embedding_only(filter_fn):
    """Filter function that targets only nn.Embedding modules."""
    return lambda mod, fqn: isinstance(mod, nn.Embedding)


# ── Quantized model variants ──────────────────────────────────────────────────

def get_model_variants(dm, cfg, path, pruned=False):
    """
    Returns a list of (label, model) tuples covering:
      fp32, fp16, int8 (bnb), int8 (torchao), int4 (torchao), nf4 (bnb)
    """
    variants = []

    def fresh(prune_emb=pruned):
        m = load_fresh_model(dm, cfg, path)
        if prune_emb:
            m.item_embedding = prune_embedding(m.item_embedding)
        return m

    # ── fp32 baseline ──────────────────────────────────────────────────────
    variants.append(("fp32", fresh()))

    # ── fp16 ──────────────────────────────────────────────────────────────
    m = fresh()
    m.item_embedding = copy.deepcopy(m.item_embedding).half()
    variants.append(("fp16", m))

    # ── int8 via bitsandbytes (reference) ─────────────────────────────────
    m = fresh()
    emb_bnb8 = Embedding8bit(m.item_embedding.weight.shape[0], m.item_embedding.weight.shape[1])
    emb_bnb8.load_state_dict(m.item_embedding.state_dict())
    m.item_embedding = emb_bnb8
    variants.append(("int8_bnb", m))

    # ── int8 via torchao ──────────────────────────────────────────────────
    m = fresh()
    quantize_(m, int8_weight_only(), filter_fn=apply_embedding_only())
    variants.append(("int8_ao", m))

    # ── int4 via torchao ──────────────────────────────────────────────────
    # group_size controls the granularity of quantization.
    # Smaller = better quality, more overhead. 32 is a good starting point.
    m = fresh()
    quantize_(m, int4_weight_only(group_size=32), filter_fn=apply_embedding_only())
    variants.append(("int4_ao", m))

    # ── nf4 via bitsandbytes (kept for comparison) ─────────────────────────
    # NF4 is not available in torchao — it is a bitsandbytes-specific format
    # designed for QLoRA training, not inference. Included here so you can
    # compare it against torchao int4 on equal footing.
    from bitsandbytes.nn import Embedding4bit
    m = fresh()
    emb_nf4 = Embedding4bit(
        m.item_embedding.weight.shape[0],
        m.item_embedding.weight.shape[1],
        quant_type="nf4",
    )
    emb_nf4.load_state_dict(m.item_embedding.state_dict())
    m.item_embedding = emb_nf4
    variants.append(("nf4_bnb", m))

    return variants


# ── Main ──────────────────────────────────────────────────────────────────────

def main(path, cfg):
    all_results = {}

    for pruned in [False, True]:
        suffix = "_pruned" if pruned else ""
        dm = DataManager(
            cfg.model.type,
            cfg.dataset.name,
            cfg.training.batch_size,
            cfg.model.params.max_sequence_length,
        )

        variants = get_model_variants(dm, cfg, path, pruned=pruned)

        for label, model in variants:
            key = label + suffix
            print(f"\n=== {key} ===")
            result = evaluate_model(model, dm)
            all_results[key] = result
            model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config_path = "configs/bert/ml-1m.yaml"
    cfg = OmegaConf.load(config_path)
    set_seed(cfg.seed)

    model_path = "trained_models/bert_model_ml-1m_42.pth"
    results = main(model_path, cfg)

    os.makedirs("per_user_results", exist_ok=True)
    base_name = model_path[15:-4]

    for key, result in results.items():
        user_df = pd.DataFrame({
            "user_id": list(result["user_hr"].keys()),
            "hr": list(result["user_hr"].values()),
            "ndcg": [result["user_ndcg"][u] for u in result["user_hr"].keys()],
        })
        user_df.to_csv(f"per_user_results/{base_name}_{key}.csv", index=False)
        print(f"Saved per_user_results/{base_name}_{key}.csv")

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nSaved results.pkl")