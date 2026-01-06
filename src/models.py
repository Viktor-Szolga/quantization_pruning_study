import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64]):
        super().__init__()
        in_sizes = [input_size] + hidden_sizes[:-1]
        self.module_list = nn.ModuleList()

        for in_size, out_size in zip(in_sizes, hidden_sizes):
            self.module_list.append(nn.Linear(in_size, out_size))
            self.module_list.append(nn.ReLU())
        
    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class NeuralMF(nn.Module):
    def __init__(self, num_users, num_items, latent_space_size, hidden_sizes=[256, 128, 64]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, latent_space_size)
        self.item_embedding = nn.Embedding(num_items, latent_space_size)
        self.mlp = MLP(latent_space_size*2, hidden_sizes)

        self.prediction_layer = nn.Linear(hidden_sizes[-1], 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_id, item_id):
        if user_id.dim() > 1:
            user_id = user_id.squeeze(1)
        if item_id.dim() > 1:
            item_id = item_id.squeeze(1)
            
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)

        vector = torch.cat([user_emb, item_emb], dim=-1)
        out = self.mlp(vector)
        logits = self.prediction_layer(out)
        return logits.squeeze()