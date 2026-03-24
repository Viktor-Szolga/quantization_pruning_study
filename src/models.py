import torch.nn as nn
import torch
import bitsandbytes as bnb

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


class NeuralCF(nn.Module):
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
    


class NeuralMF(nn.Module):
    def __init__(self, num_users, num_items, latent_mf, latent_mlp, hidden_sizes=[256, 128, 64]):
        super().__init__()
        self.user_embedding_mf = nn.Embedding(num_users, latent_mf)
        self.item_embedding_mf = nn.Embedding(num_items, latent_mf)
        self.user_embedding_mlp = nn.Embedding(num_users, latent_mlp)
        self.item_embedding_mlp = nn.Embedding(num_items, latent_mlp)

        self.mlp = MLP(latent_mlp*2, hidden_sizes)
        self.prediction_layer = nn.Linear(hidden_sizes[-1] + latent_mf, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding_mf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
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

        user_emb_mf = self.user_embedding_mf(user_id)
        item_emb_mf = self.item_embedding_mf(item_id)
        user_emb_mlp = self.user_embedding_mlp(user_id)
        item_emb_mlp = self.item_embedding_mlp(item_id)


        vector_mf = user_emb_mf * item_emb_mf
        vector_mlp = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        out = self.mlp(vector_mlp)
        logits = self.prediction_layer(torch.cat([vector_mf, out], dim=-1))
        score = self.sigmoid(logits)
        return score
    


class Bert4Rec(nn.Module):
    def __init__(self, item_num, hidden_size, num_layers, num_heads, max_sequence_length, hidden_dropout=0.5, attention_dropout=0.2):
        super().__init__()
        self.item_embedding = nn.Embedding(item_num + 2, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True, dim_feedforward=hidden_size * 4, dropout=attention_dropout, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_size, item_num + 2)
        
        self.apply(self._init_weights)
        self.item_embedding.weight.data[0].fill_(0)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Standard BERT initialization
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        mask = (x == 0)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.item_embedding(x) + self.position_embedding(positions)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.output_layer(x)


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim_mf, latent_dim_mlp, hidden_sizes=[64, 32, 16], dropout_prob=0.5):
        super(NeuMF, self).__init__()
        self.embed_user_GMF = nn.Embedding(num_users, latent_dim_mf)
        self.embed_item_GMF = nn.Embedding(num_items, latent_dim_mf)

        self.embed_user_MLP = nn.Embedding(num_users, latent_dim_mlp)
        self.embed_item_MLP = nn.Embedding(num_items, latent_dim_mlp)
        
        input_size = latent_dim_mlp * 2
        modules = []
        for h_size in hidden_sizes:
            modules.append(nn.Linear(input_size, h_size))
            modules.append(nn.Dropout(dropout_prob))
            modules.append(nn.ReLU())
            input_size = h_size
        self.mlp_network = nn.Sequential(*modules)

        self.prediction_layer = nn.Linear(latent_dim_mf + hidden_sizes[-1], 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.mlp_network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        nn.init.xavier_uniform_(self.prediction_layer.weight)
        if self.prediction_layer.bias is not None:
            nn.init.zeros_(self.prediction_layer.bias)

    def forward(self, user_indices, item_indices):
        user_gmf = self.embed_user_GMF(user_indices)
        item_gmf = self.embed_item_GMF(item_indices)
        mf_vector = torch.mul(user_gmf, item_gmf)

        user_mlp = self.embed_user_MLP(user_indices)
        item_mlp = self.embed_item_MLP(item_indices)
        mlp_vector = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_vector = self.mlp_network(mlp_vector)

        combined_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        
        prediction = self.prediction_layer(combined_vector)
        return prediction.view(-1)
    




class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super().__init__()
        self.embed_user = nn.Embedding(num_users + 1, latent_dim)
        self.embed_item = nn.Embedding(num_items + 1, latent_dim)
        self.prediction = nn.Linear(latent_dim, 1)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.kaiming_uniform_(self.prediction.weight, a=1, nonlinearity='sigmoid')

    def forward(self, users, items):
        u = self.embed_user(users)
        i = self.embed_item(items)
        vec = u * i               # element-wise multiply
        return self.prediction(vec).view(-1)
    

class MLP(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, hidden_sizes=[128,64]):
        super().__init__()
        self.embed_user = nn.Embedding(num_users + 1, latent_dim)
        self.embed_item = nn.Embedding(num_items + 1, latent_dim)

        input_size = latent_dim * 2
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))   # optional
            input_size = h
        self.mlp = nn.Sequential(*layers)
        self.prediction = nn.Linear(hidden_sizes[-1], 1)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.prediction.weight, a=1, nonlinearity='sigmoid')

    def forward(self, users, items):
        u = self.embed_user(users)
        i = self.embed_item(items)
        x = torch.cat([u, i], dim=-1)
        x = self.mlp(x)
        return self.prediction(x).view(-1)