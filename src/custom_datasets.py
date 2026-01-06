from torch.utils.data import Dataset
import torch

class NMFDatset(Dataset):
    def __init__(self, data_array):
        self.users = torch.LongTensor(data_array[:, 0])
        self.items = torch.LongTensor(data_array[:, 1])
        self.ratings = torch.FloatTensor(data_array[:, 2])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class BERTDataset(Dataset):
    def __init__(self, user_sequences, max_len=50, num_items=3706):
        self.user_ids = list(user_sequences.keys())
        self.sequences = list(user_sequences.values())
        self.max_len = max_len
        self.num_items = num_items

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        padded_seq = [0] * padding_len + seq
        return torch.LongTensor(padded_seq)