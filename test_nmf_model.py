import torch
import numpy as np
import random
from src.data_manager import MovieLensDataManager
from src.trainer import RecSysTrainer
from src.models import NeuralMF
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    set_seed(42)

    model_type = "nmf"
    data_manager = MovieLensDataManager(model_type)
    model = NeuralMF(num_users=data_manager.num_users + 1, num_items=data_manager.num_items + 1, latent_mf=4, latent_mlp=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.load_state_dict(torch.load("testing/best_nmf_model.pth", map_location="cuda"))
    trainer = RecSysTrainer(model, optimizer, criterion)
    

    hit_rate, ndcg = trainer.evaluate(data_manager.valid_loader)
    print(hit_rate)
    print(ndcg)