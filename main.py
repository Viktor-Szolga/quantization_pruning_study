import torch
import numpy as np
import random
from src.data_manager import MovieLensDataManager
from src.trainer import RecSysTrainer
from src.models import NeuralMF
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False






if __name__ == "__main__":
    set_seed(42)

    data_manager = MovieLensDataManager("nmf")
    model = NeuralMF(num_users=data_manager.num_users, num_items=data_manager.num_items, latent_space_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    trainer = RecSysTrainer(model, optimizer, criterion)

    epochs = 10
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        train_losses.append(trainer.train_epoch(data_manager.train_loader, data_manager.num_items))
        valid_losses.append(trainer.evaluate(data_manager.valid_loader))

    plt.plot(range(epochs), train_losses, label="Train")
    plt.show()