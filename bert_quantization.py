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
from src.models import Bert4Rec
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
        input = next(iter(dm.valid_loader))[0].to(device)
        _ = model(input)
    print(f"Memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
    print(f"Peak memory used: {torch.cuda.max_memory_allocated(device)/1024**2:.2f} MB")


def evaluate_model(model, dm, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Number of neurons in embedding: {m.item_embedding.weight.numel()}")
    trainer = RecSysTrainer(model, None, None, device)
    with torch.amp.autocast("cuda"):
        hr, ndcg = trainer.evaluate(dm.valid_loader)
        
    print(model.item_embedding.weight.dtype)
    print(f"NDCG: {ndcg:.4f} | HR: {hr:.4f}")
    print_size(model, dm)
    print(f"---"*10)


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
    for value in quantized_list:
        m = copy.deepcopy(model)
        setattr(m, named_parameter, value)
        variants.append(m)
    return variants


if __name__ == "__main__":
    prune_model = [False, True]
    for p in prune_model:
        dm = MovieLensDataManager("bert")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load base model structure
        model = Bert4Rec(dm.num_items, 128, 8, 4, dm.train_set.max_len)
        model.load_state_dict(torch.load("testing/best_bert_model.pth"))
        trained_embedding = model.item_embedding
        if p:
            trained_embedding = prune_embedding(trained_embedding)


        test_embedding = nn.Embedding(100000, 10000)
        test_embeddings = get_quantized_embeddings(test_embedding)
        models = get_model_variants(model, get_quantized_embeddings(trained_embedding), "item_embedding")

        for m, test_embedding in zip(models, test_embeddings):
            m.test_embedding = test_embedding
            evaluate_model(m, dm)
            m.to("cpu")

            gc.collect()
            torch.cuda.empty_cache()