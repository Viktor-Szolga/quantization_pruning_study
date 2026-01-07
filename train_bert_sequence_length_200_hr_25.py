import torch
import numpy as np
import random
from src.data_manager import MovieLensDataManager
from src.trainer import RecSysTrainer
from src.models import Bert4Rec
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import get_linear_schedule_with_warmup


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)
    device="cuda" if torch.cuda.is_available() else "cpu"
    epochs = 300
    
    print(f"Running on {device}")
    model_type = "bert"
    data_manager = MovieLensDataManager(model_type)
    model = Bert4Rec(item_num=data_manager.num_items, hidden_size=128, num_layers=8, num_heads=4,
                    max_sequence_length=data_manager.train_set.max_len, dropout=0.3
                )
    optimizer = torch.optim.AdamW(
                                model.parameters(),
                                lr=1e-3,
                                weight_decay=0.01
                            )
    
    num_training_steps = epochs * len(data_manager.train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
                                            optimizer,
                                            num_warmup_steps=num_warmup_steps,
                                            num_training_steps=num_training_steps
                                        )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    trainer = RecSysTrainer(model, optimizer, criterion, device=device, scheduler=scheduler)
    
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