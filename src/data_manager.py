import pickle
from src.custom_datasets import NMFDatset, BERTDataset
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

class MovieLensDataManager:
    def __init__(self, model_type: str):
        with open(ROOT_DIR / "data" / "processed" / model_type.lower() / "train.pkl", "rb") as f:
            self.train_data = pickle.load(f)
        with open(ROOT_DIR / "data" / "processed" / model_type.lower() / "valid.pkl", "rb") as f:
            self.valid_data = pickle.load(f)
        with open(ROOT_DIR / "data" / "processed" / model_type.lower() / "test.pkl", "rb") as f:
            self.test_data = pickle.load(f)


        match model_type.upper():
            case "NMF":
                self.train_set = NMFDatset(self.train_data)
                self.valid_set = NMFDatset(self.valid_data)
                self.test_set = NMFDatset(self.test_data)
                
                self.num_users = len(np.unique(self.train_data[:, 0]))
                self.num_items = len(np.unique(self.train_data[:, 1]))
                    
            case "BERT":
                self.train_set = BERTDataset(self.train_data)
                self.valid_set = BERTDataset(self.valid_data)
                self.test_set = BERTDataset(self.test_data)

        self.train_loader = DataLoader(self.train_set, batch_size=256, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=256, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=256, shuffle=False)


    