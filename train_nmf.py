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
    trainer = RecSysTrainer(model, optimizer, criterion)
    epochs = 500
    train_losses = []
    ndcg_list = []
    hit_list = []

    best_ndcg = float("-inf")
    for epoch in tqdm(range(epochs), desc="Training", total=epochs):
        train_losses.append(trainer.train_epoch(data_manager.train_loader, data_manager.num_items))
        hr, ndcg = trainer.evaluate(data_manager.valid_loader)
        ndcg_list.append(ndcg)
        hit_list.append(hr)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            save_path = Path("trained_models") / f"best_{model_type}_model.pth"
            torch.save(model.state_dict(), str(save_path))

    plt.plot(range(epochs), train_losses, label="Train")
    plt.title("Train loss")
    plt.show()

    plt.plot(range(epochs), hit_list, label="Train")
    plt.title("Hr")
    plt.show()

    plt.plot(range(epochs), ndcg_list, label="Train")
    plt.title("NDCG")
    plt.show()