from torch.utils.data import Dataset
import torch
import random
import numpy as np

class NMFDatset(Dataset):
    def __init__(self, data_array, num_items, user_item_set, num_negatives, 
                 pop_prob=None, smooth_popularity=False):
        self.users = torch.LongTensor(data_array[:, 0].copy())
        self.items = torch.LongTensor(data_array[:, 1].copy())
        self.ratings = torch.FloatTensor(data_array[:, 2].copy())
        self.num_negatives = num_negatives
        self.item_ids = np.arange(1, num_items + 1)
        self.user_item_set = user_item_set

        if pop_prob is not None:
            pop_prob = np.array(pop_prob)
            self.pop_prob = pop_prob[1:]
            if smooth_popularity:
                self.pop_prob = self.pop_prob ** 0.75
            self.pop_prob = self.pop_prob / self.pop_prob.sum()
        else:
            self.pop_prob = np.ones(len(self.item_ids)) / len(self.item_ids)

    def __len__(self):
        return len(self.users)

    def _sample_negatives(self, user, pos_item):
        user_history = self.user_item_set[user.item()]  # set of interacted items

        mask = np.ones(len(self.item_ids), dtype=bool)

        exclude_items = set(user_history)
        exclude_items.add(pos_item.item())

        exclude_items = np.array(list(exclude_items))
        exclude_idx = np.isin(self.item_ids, exclude_items)

        mask[exclude_idx] = False

        candidates = self.item_ids[mask]
        probs = self.pop_prob[mask]
        probs = probs / probs.sum()

        return np.random.choice(
            candidates,
            size=self.num_negatives,
            replace=False,
            p=probs
        )
    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]

        if self.num_negatives > 0:
            neg_items = self._sample_negatives(user, pos_item)
            items = torch.LongTensor([pos_item] + neg_items.tolist())
            return user, items, self.ratings[idx]

        return self.users[idx], self.items[idx], self.ratings[idx]

class BERTDataset(Dataset):
    def __init__(self, data, prob, num_items, max_len=20, mask_prob=0.2, mode='train',
                num_negatives=100):
        self.data = data[0]
        self.users = data[1]

        self.max_len = max_len
        self.num_items = num_items
        self.padding_token = 0
        self.mask_token = num_items + 1
        self.mask_prob = mask_prob
        self.mode = mode

        self.num_negatives = num_negatives

        self.pop_prob = prob[1:]
        self.pop_prob = self.pop_prob / self.pop_prob.sum()
        self.item_ids = np.arange(1, num_items + 1)
        self.streaming = len(self.users) > 100000 and mode == 'train' # Helps with data augmentation on large datasets
        
        if not self.streaming:
            self.samples = []
            for user in self.users:
                seq = self.data[user]['seq']

                if len(seq) < 2:
                    continue

                if mode == 'train':
                    for end in range(2, len(seq) + 1):
                        sub_seq = seq[:end]
                        self.samples.append((user, sub_seq))
                else:
                    self.samples.append((user, seq))
        else:
            self.user_offsets = []
            self.total_len = 0

            for user in self.users:
                seq_len = len(self.data[user]['seq'])

                if seq_len < 2:
                    continue

                if self.mode == 'train':
                    n = seq_len - 1
                else:
                    n = 1

                self.user_offsets.append((user, self.total_len, self.total_len + n))
                self.total_len += n

    def __len__(self):
        if not self.streaming:
            return len(self.samples)
        else:
            return self.total_len
    def _sample_negative(self, exclude_set):
        mask = np.ones(self.num_items, dtype=bool)

        exclude = np.array(list(exclude_set)) - 1
        mask[exclude] = False

        probs = self.pop_prob[mask]
        probs = probs / probs.sum()

        candidates = self.item_ids[mask]

        return np.random.choice(
            candidates,
            size=self.num_negatives,
            replace=False,
            p=probs
        )

    def __getitem__(self, idx):
        if not self.streaming:
            user, seq = self.samples[idx]
        else:
            left, right = 0, len(self.user_offsets) - 1
            while left <= right:
                mid = (left + right) // 2
                user_mid, start, end = self.user_offsets[mid]

                if start <= idx < end:
                    break
                elif idx < start:
                    right = mid - 1
                else:
                    left = mid + 1

            user, start, end = self.user_offsets[mid]
            seq_full = self.data[user]['seq']

            if self.mode == 'train':
                pos = idx - start + 2
                seq = seq_full[:pos]
            else:
                seq = seq_full

        if self.mode in ['valid', 'test']:
            target = self.data[user]["target"]
            if isinstance(target, (list, np.ndarray)):
                target = target[0]
            target = int(target)

            user_items = set(seq)
            user_items.add(target)

            neg_items = self._sample_negative(user_items)

            items = torch.LongTensor([target] + neg_items.tolist())

            tokens = seq + [self.mask_token]
            tokens = tokens[-self.max_len:]

            padding_len = self.max_len - len(tokens)
            tokens = [self.padding_token] * padding_len + tokens

            return torch.LongTensor(tokens), items, torch.LongTensor([user])

        else:
            seq = seq[-self.max_len:]
            padding_len = self.max_len - len(seq)

            seq = [self.padding_token] * padding_len + seq

            tokens = []
            labels = []

            num_masked = 0

            for s in seq:
                if s == 0:
                    tokens.append(0)
                    labels.append(0)
                    continue

                prob = random.random()

                if prob < self.mask_prob:
                    num_masked += 1
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)

                    elif prob < 0.9:
                        #-----------------Changed------------------
                        #neg = np.random.choice(self.item_ids, p=self.pop_prob)
                        neg = self._sample_negative(set(seq))[0]
                        #------------------End Changed-------------
                        tokens.append(int(neg))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

            if num_masked == 0:
                valid_indices = [i for i, s in enumerate(seq) if s != 0]
                if len(valid_indices) > 0:
                    i = random.choice(valid_indices)
                    labels[i] = seq[i]
                    tokens[i] = self.mask_token

            return (
                torch.LongTensor(tokens),
                torch.LongTensor(labels),
                torch.LongTensor([user])
            )