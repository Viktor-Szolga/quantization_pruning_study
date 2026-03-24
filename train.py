import torch
import numpy as np
import random
from src.data_manager import DataManager
from src.trainer import RecSysTrainer
from src.models import NeuMF, Bert4Rec
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import get_linear_schedule_with_warmup
import os
from omegaconf import OmegaConf
from src.utils import set_seed
import argparse    
import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors")

def main(config_path, seed):
    cfg = OmegaConf.load(config_path)
    set_seed(seed)
    device = "cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else cfg.device

    data_manager = DataManager(cfg.model.type, cfg.dataset.name, cfg, cfg.training.batch_size, cfg.model.params.get("max_sequence_length", 0), smooth_popularity=cfg.training.get("smooth_popularity", False))
    match cfg.model.type:
        case "nmf":
            model = NeuMF(num_users=data_manager.num_users + 1, num_items=data_manager.num_items + 1,
                           latent_dim_mf=cfg.model.params.latent_mf, latent_dim_mlp=cfg.model.params.latent_mlp,
                             hidden_sizes=cfg.model.params.hidden_sizes, dropout_prob=cfg.model.params.dropout_rate)
        case "bert":
            model = Bert4Rec(item_num=data_manager.num_items, hidden_size=cfg.model.params.hidden_size, num_layers=cfg.model.params.num_layers, num_heads=cfg.model.params.num_heads,
                    max_sequence_length=cfg.model.params.max_sequence_length, hidden_dropout=cfg.model.params.hidden_dropout, attention_dropout=cfg.model.params.attention_dropout)

    if cfg.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    
    match cfg.loss.name:
        case "BCELossLogits":
            criterion = torch.nn.BCEWithLogitsLoss()
        case "CrossEntropyLoss":
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    match cfg.scheduler.name:
        case "linear_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                                            optimizer,
                                            num_warmup_steps=cfg.training.update_steps*cfg.training.warmup_ratio,
                                            num_training_steps=cfg.training.update_steps
                                        )
        case "cosineAnnealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                            optimizer,
                                            T_max=cfg.training.max_steps,
                                            eta_min=1e-5
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
            os.makedirs(cfg.saving.save_dir, exist_ok=True)
            print(trainer.model)
            train_losses, hit_list, ndcg_list, eval_at = trainer.train_n_steps_nmf(
                data_manager.train_loader, data_manager.valid_loader, data_manager.num_items, cfg, 
                item_popularity=torch.tensor(data_manager.popularity["prob"], dtype=torch.float32),
                save_path=f"{cfg.saving.save_dir}/{cfg.saving.filename}_{cfg.dataset.name}_{seed}",
                max_norm=cfg.training.max_norm)

            os.makedirs(f"{cfg.saving.figure_dir}/{config_path[8:-5]}", exist_ok=True)

            plt.plot(range(eval_at[-1]), train_losses, label="Train")
            plt.title("Train loss")
            plt.savefig(f"{cfg.saving.figure_dir}/{config_path[8:-5]}/train_loss_{seed}.png")
            plt.close()

            plt.plot(eval_at, hit_list, label="HR")
            plt.title("HR")
            plt.savefig(f"{cfg.saving.figure_dir}/{config_path[8:-5]}/hit_rate_{seed}.png")
            plt.close()

            plt.plot(eval_at, ndcg_list, label="NDCG")
            plt.title("NDCG")
            plt.savefig(f"{cfg.saving.figure_dir}/{config_path[8:-5]}/ndcg_{seed}.png")
            plt.close()
        case "bert":
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
            trainer = RecSysTrainer(model, optimizer, criterion, device=device, scheduler=scheduler)
            
            train_losses = []
            ndcg_list = []
            hit_list = []
            
            os.makedirs(cfg.saving.save_dir, exist_ok=True)
            train_losses, hit_list, ndcg_list, eval_at = trainer.train_n_steps_bert(data_manager.train_loader, data_manager.valid_loader, accumulation_steps=cfg.training.accumulation_steps, validation_interval=cfg.evaluation.interval,
                                                                                     max_steps=cfg.training.accumulation_steps*cfg.training.update_steps, save_path=f"{cfg.saving.save_dir}/{cfg.saving.filename}_{cfg.dataset.name}_{seed}", cfg=cfg)

            os.makedirs(f"{cfg.saving.figure_dir}/{config_path[8:-5]}", exist_ok=True)

            plt.plot(range(eval_at[-1]), train_losses, label="Train")
            plt.title("Train loss")
            plt.savefig(f"{cfg.saving.figure_dir}/{config_path[8:-5]}/train_loss_{seed}.png")
            plt.close()

            plt.plot(eval_at, hit_list, label="HR")
            plt.title("HR")
            plt.savefig(f"{cfg.saving.figure_dir}/{config_path[8:-5]}/hit_rate_{seed}.png")
            plt.close()

            plt.plot(eval_at, ndcg_list, label="NDCG")
            plt.title("NDCG")
            plt.savefig(f"{cfg.saving.figure_dir}/{config_path[8:-5]}/ndcg_{seed}.png")
            plt.close()
            print(max(ndcg_list))
    print("Performance on test:")
    hr, ndcg = trainer.evaluate(data_manager.test_loader)
    print(f" Test loader hr: {hr} | ndcg: {ndcg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="ml-1m",
        help="Dataset name (one of 'ml-1m', 'ml-20m', 'beauty', 'steam')"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="nmf",
        help="Model type (one of 'bert', 'nmf')"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Model type (one of 'bert', 'nmf')"
    )
    args = parser.parse_args()
    main(f"configs/{args.model}/{args.dataset}.yaml", args.seed)