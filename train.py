import torch
import numpy as np
import random
from src.data_manager import MovieLensDataManager
from src.trainer import RecSysTrainer
from src.models import NeuralMF, Bert4Rec
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import get_linear_schedule_with_warmup
import os
from omegaconf import OmegaConf
from src.utils import set_seed    

def main(config_path):
    cfg = OmegaConf.load(config_path)
    set_seed(cfg.seed)
    device = "cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else cfg.device

    data_manager = MovieLensDataManager(cfg.model.type, cfg.dataset.name)
    match cfg.model.type:
        case "nmf":
            model = NeuralMF(num_users=data_manager.num_users + 1, num_items=data_manager.num_items + 1, latent_mf=cfg.model.params.latent_mf, latent_mlp=cfg.model.params.latent_mlp, hidden_sizes=cfg.model.params.hidden_sizes)
        case "bert":
            model = Bert4Rec(item_num=data_manager.num_items, hidden_size=cfg.model.params.hidden_size, num_layers=cfg.model.params.num_layers, num_heads=cfg.model.params.num_heads,
                    max_sequence_length=data_manager.train_set.max_len, dropout=cfg.model.params.dropout
                )

    if cfg.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    
    match cfg.loss.name:
        case "BCELoss":
            criterion = torch.nn.BCELoss()
        case "CrossEntropyLoss":
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    match cfg.scheduler.name:
        case "linear_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                                            optimizer,
                                            num_warmup_steps=cfg.training.max_steps*cfg.training.warmup_ratio,
                                            num_training_steps=cfg.training.max_steps
                                        )
        case None:
            scheduler = None

    trainer = RecSysTrainer(model, optimizer, criterion, device=device)

    train_losses = []
    ndcg_list = []
    hit_list = []

    best_ndcg = float("-inf")

    match cfg.model.type:
        case "nmf":
            for epoch in tqdm(range(cfg.training.epochs), desc="Training", total=cfg.training.epochs):
                train_losses.append(trainer.train_epoch_nmf(data_manager.train_loader, data_manager.num_items))
                hr, ndcg = trainer.evaluate(data_manager.valid_loader)
                ndcg_list.append(ndcg)
                hit_list.append(hr)
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    os.makedirs(cfg.saving.save_dir, exist_ok=True)
                    save_path = Path(cfg.saving.save_dir) / f"{cfg.saving.filename}_{cfg.seed}"
                    torch.save(model.state_dict(), str(save_path))

            os.makedirs(f"{cfg.saving.figure_dir}/{config_path[8:]}_{cfg.seed}", exist_ok=True)

            plt.plot(range(cfg.training.epochs), train_losses, label="Train")
            plt.title("Train loss")
            plt.savefig(f"{cfg.saving.figure_dir}/{config_path[8:]}_{cfg.seed}/train_loss.png")
            plt.close()

            plt.plot(range(cfg.training.epochs), hit_list, label="HR")
            plt.title("HR")
            plt.savefig(f"{cfg.saving.figure_dir}/{config_path[8:]}_{cfg.seed}/hit_rate.png")
            plt.close()

            plt.plot(range(cfg.training.epochs), ndcg_list, label="NDCG")
            plt.title("NDCG")
            plt.savefig(f"{cfg.saving.figure_dir}/{config_path[8:]}_{cfg.seed}/ndcg.png")
            plt.close()
        case "bert":
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
            
            train_losses, hit_list, ndcg_list, eval_at = trainer.train_n_steps_bert(data_manager.train_loader, data_manager.valid_loader, max_steps=num_training_steps, save_path=f"{cfg.saving.save_dir}/{cfg.saving.filename}_{cfg.seed}")

            os.makedirs(f"{cfg.saving.figure_dir}/{config_path[8:]}", exist_ok=True)

            plt.plot(range(cfg.training.max_steps), train_losses, label="Train")
            plt.title("Train loss")
            plt.savefig(f"{cfg.saving.figure_dir}/{config_path[8:]}_{cfg.seed}/train_loss.png")
            plt.close()

            plt.plot(range(cfg.training.max_steps), hit_list, label="HR")
            plt.title("HR")
            plt.savefig(f"{cfg.saving.figure_dir}/{config_path[8:]}_{cfg.seed}/hit_rate.png")
            plt.close()

            plt.plot(range(cfg.training.max_steps), ndcg_list, label="NDCG")
            plt.title("NDCG")
            plt.savefig(f"{cfg.saving.figure_dir}/{config_path[8:]}_{cfg.seed}/ndcg.png")
            plt.close()
    

if __name__ == "__main__":
    main("configs/bert/ml-1m.yaml")