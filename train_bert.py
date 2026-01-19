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
    data_manager = MovieLensDataManager(model_type, dataset="ml-1m")
    model = Bert4Rec(item_num=data_manager.num_items, hidden_size=256, num_layers=2, num_heads=8,
                    max_sequence_length=data_manager.train_set.max_len, dropout=0.2
                )
    optimizer = torch.optim.AdamW(
                                model.parameters(),
                                lr=1e-3,
                                weight_decay=0.01
                            )
    
    num_training_steps = 1000
    num_training_steps = 10_000
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
    
    train_losses, hit_list, ndcg_list, eval_at = trainer.train_n_steps(data_manager.train_loader, data_manager.valid_loader, max_steps=num_training_steps)

    plt.plot(range(num_training_steps), train_losses, label="Train")
    plt.title("Train loss")
    plt.show()

    plt.plot(eval_at, hit_list, label="Valid HR")
    plt.title("Hr")
    plt.show()

    plt.plot(eval_at, ndcg_list, label="Valid NDCG")
    plt.title("NDCG")
    plt.show()