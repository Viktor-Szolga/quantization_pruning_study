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


dm = MovieLensDataManager("bert")
# Load base model structure
model = Bert4Rec(dm.num_items, 128, 8, 4, dm.train_set.max_len)
model.load_state_dict(torch.load("testing/best_bert_model.pth"))

# Load FP32/FP16 model
model.load_state_dict(torch.load("model_fp32.pth"))

# Quantize for inference
model_int8 = bnb.nn.quantize(model, dtype=torch.int8)
model_int4 = bnb.nn.quantize(model, dtype="nf4")


print(model_int8)
print(model_int4)