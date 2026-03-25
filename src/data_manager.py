import pickle
from src.custom_datasets import NMFDatset, BERTDataset
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent

class DataManager:
    def __init__(self, model_type, dataset, cfg, batch_size=128, max_sequence_length=100, smooth_popularity=False):
        with open(ROOT_DIR / "data" / f"processed_{dataset}" / model_type.lower() / "train.pkl", "rb") as f:
            self.train_data = pickle.load(f)
        with open(ROOT_DIR / "data" / f"processed_{dataset}" / model_type.lower() / "valid.pkl", "rb") as f:
            self.valid_data = pickle.load(f)
        with open(ROOT_DIR / "data" / f"processed_{dataset}" / model_type.lower() / "test.pkl", "rb") as f:
            self.test_data = pickle.load(f)
        with open(ROOT_DIR / "data" / f"processed_{dataset}" / "stats.pkl", "rb") as f:
            self.num_users, self.num_items = pickle.load(f)
        with open(ROOT_DIR / "data" / f"processed_{dataset}" / "popularity.pkl", "rb") as f:
            self.popularity = pickle.load(f)
            
        match model_type.upper():
            case "NMF":
                self.train_set = NMFDatset(self.train_data, num_negatives=cfg.training.num_negatives, user_item_set=self.popularity["user_item_set"], pop_prob=self.popularity["prob"], smooth_popularity=smooth_popularity, num_items=self.num_items)
                self.valid_set = NMFDatset(self.valid_data, num_negatives=100, user_item_set=self.popularity["user_item_set"], pop_prob=None, num_items=self.num_items)
                self.test_set = NMFDatset(self.test_data, num_negatives=100, user_item_set=self.popularity["user_item_set"], pop_prob=None, num_items=self.num_items)
            case "BERT":
                self.train_set = BERTDataset(self.train_data, prob=self.popularity["prob"], num_items=self.num_items, mode="train", max_len=max_sequence_length)
                self.valid_set = BERTDataset(self.valid_data, prob=self.popularity["prob"], num_items=self.num_items, mode="valid", num_negatives=100, max_len=max_sequence_length)
                self.test_set = BERTDataset(self.test_data, prob=self.popularity["prob"], num_items=self.num_items, mode="test", num_negatives=100, max_len=max_sequence_length)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False)