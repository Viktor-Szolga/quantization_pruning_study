import pickle
from src.custom_datasets import NMFDatset, BERTDataset
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent

class MovieLensDataManager:
    def __init__(self, model_type: str, dataset="ml-1m"):
        if dataset == "ml-1m":
            processing_dir = "processed"
        if dataset == "ml-20m":
            processing_dir = "processed20"
        with open(ROOT_DIR / "data" / processing_dir / model_type.lower() / "train.pkl", "rb") as f:
            self.train_data = pickle.load(f)
        with open(ROOT_DIR / "data" / processing_dir / model_type.lower() / "valid.pkl", "rb") as f:
            self.valid_data = pickle.load(f)
        with open(ROOT_DIR / "data" / processing_dir / model_type.lower() / "test.pkl", "rb") as f:
            self.test_data = pickle.load(f)

        if dataset == "ml-1m":
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
        if dataset == "ml-20m":
            self.num_users = 138493
            self.num_items = 18345   # Has to be changed manually atm
        match model_type.upper():
            case "NMF":
                self.train_set = NMFDatset(self.train_data)
                self.valid_set = NMFDatset(self.valid_data, all_item_ids=np.arange(1, self.num_items+1), num_negatives=99)
                self.test_set = NMFDatset(self.test_data, all_item_ids=np.arange(1, self.num_items+1), num_negatives=99)
            case "BERT":
                self.train_set = BERTDataset(self.train_data, mode="train")
                self.valid_set = BERTDataset(self.valid_data, all_item_ids=np.arange(1, self.num_items+1), mode="valid", num_negatives=99)
                self.test_set = BERTDataset(self.test_data, all_item_ids=np.arange(1, self.num_items+1), mode="test", num_negatives=99)

        self.train_loader = DataLoader(self.train_set, batch_size=256, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=256, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=256, shuffle=False)


    