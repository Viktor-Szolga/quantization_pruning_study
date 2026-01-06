import pickle
from src.custom_datasets import NMFDatset, BERTDataset
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent

class MovieLensDataManager:
    def __init__(self, model_type: str):
        with open(ROOT_DIR / "data" / "processed" / model_type.lower() / "train.pkl", "rb") as f:
            self.train_data = pickle.load(f)
        with open(ROOT_DIR / "data" / "processed" / model_type.lower() / "valid.pkl", "rb") as f:
            self.valid_data = pickle.load(f)
        with open(ROOT_DIR / "data" / "processed" / model_type.lower() / "test.pkl", "rb") as f:
            self.test_data = pickle.load(f)

        self.num_items =  pd.read_csv(
                            ROOT_DIR / "data" / "ml-1m" / "movies.dat",
                            sep="::",
                            header=None,
                            engine="python",
                            names=["MovieID", "Title", "Genres"],
                            encoding="latin-1"
                        )["MovieID"].max()
        self.num_users = pd.read_csv(
                            ROOT_DIR / "data" / "ml-1m" / "users.dat",
                            sep="::",
                            header=None,
                            engine="python",
                            names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
                            encoding="latin-1"
                        )["UserID"].max()

        match model_type.upper():
            case "NMF":
                self.train_set = NMFDatset(self.train_data)
                self.valid_set = NMFDatset(self.valid_data, all_item_ids=np.arange(1, self.num_items+1), num_negatives=99)
                self.test_set = NMFDatset(self.test_data, all_item_ids=np.arange(1, self.num_items+1), num_negatives=99)
            case "BERT":
                self.train_set = BERTDataset(self.train_data, mode="train")
                self.valid_set = BERTDataset(self.valid_data, mode="valid")
                self.test_set = BERTDataset(self.test_data, mode="test")

        self.train_loader = DataLoader(self.train_set, batch_size=256, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=256, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=256, shuffle=False)


    