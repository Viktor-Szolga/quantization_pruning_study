from torch.utils.data import Dataset
import torch
import random
import numpy as np

class NMFDatset(Dataset):
    def __init__(self, data_array, all_item_ids=None, num_negatives=0):
        self.users = torch.LongTensor(data_array[:, 0])
        self.items = torch.LongTensor(data_array[:, 1])
        self.ratings = torch.FloatTensor(data_array[:, 2])
        self.all_item_ids = all_item_ids
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]
        if self.num_negatives > 0:
            neg_items = np.random.choice(self.all_item_ids, self.num_negatives, replace=False)
            items = torch.LongTensor([pos_item] + neg_items.tolist())
            return user, items, self.ratings[idx]
        return self.users[idx], self.items[idx], self.ratings[idx]


class BERTDataset(Dataset):
    def __init__(self, data, max_len=50, num_items=3706, mask_prob=0.15, mode='train'):
        """
        data: List of dicts like [{'seq': [...], 'target': 1907}, ...]
        mode: 'train', 'valid', or 'test'
        """
        self.data = data
        self.max_len = max_len
        self.num_items = num_items
        self.mask_token = num_items + 1  # Standard Bert4Rec: item_num + 1
        self.mask_prob = mask_prob
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item['seq']
        
        if self.mode in ['valid', 'test']:
            # For LOO evaluation: 
            # 1. Take the sequence
            # 2. Append [MASK] at the end
            # 3. Target is the 'target' field from your data
            tokens = seq + [self.mask_token]
            tokens = tokens[-self.max_len:]
            
            padding_len = self.max_len - len(tokens)
            tokens = [0] * padding_len + tokens
            
            # The label is the target item ID
            target = item.get('target', 0)
            return torch.LongTensor(tokens), torch.LongTensor([target])

        else:
            # Training Mode: Cloze Task (Random Masking)
            seq = seq[-self.max_len:]
            tokens, labels = [], []

            for s in seq:
                prob = random.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob
                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(random.randint(1, self.num_items))
                    else:
                        tokens.append(s)
                    labels.append(s) # Predict the original item
                else:
                    tokens.append(s)
                    labels.append(0) # 0 means ignore in Loss Function

            # Padding
            padding_len = self.max_len - len(tokens)
            tokens = [0] * padding_len + tokens
            labels = [0] * padding_len + labels
            
            return torch.LongTensor(tokens), torch.LongTensor(labels)