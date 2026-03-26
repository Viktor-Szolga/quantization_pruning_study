import random
import numpy as np
import torch
import os
import torch.nn as nn
import gc
import torch.nn.functional as F
import psutil
import time
import pynvml
import bitsandbytes as bnb


class CUDAMemoryMonitor:
    def __init__(self, device="cuda"):
        self.device = device
        self.keep_running = True
        self.peak_vram = 0.0
        self.mean_vram = 0.0
        self.n = 0
        self.baseline_vram = 0.0

    def calibrate(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        
        time.sleep(0.1)
        self.baseline_vram = torch.cuda.memory_allocated(self.device) / 1024**2
        
        self.peak_vram = 0.0
        self.mean_vram = 0.0
        self.n = 0
        self.keep_running = True

    def measure(self):
        while self.keep_running:
            try:
                current_total = torch.cuda.memory_allocated(self.device) / 1024**2
                current_net = max(0.0, current_total - self.baseline_vram)
                
                self.n += 1
                self.mean_vram += (current_net - self.mean_vram) / self.n
                
                native_peak = torch.cuda.max_memory_allocated(self.device) / 1024**2
                self.peak_vram = max(0.0, native_peak - self.baseline_vram)
                
                time.sleep(0.005)
            except Exception: 
                break
class CUDAEnergyMonitor:
    def __init__(self, device_index=0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.keep_running = True
        self.total_energy_mj = 0.0
        self._last_time = None
        self._last_power = None

    def calibrate(self):
        self.keep_running = True
        self.total_energy_mj = 0.0
        self._last_time = None
        self._last_power = None

    def measure(self):
        while self.keep_running:
            try:
                now = time.perf_counter()
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                
                if self._last_time is not None:
                    dt = now - self._last_time
                    avg_power = (power_mw + self._last_power) / 2
                    self.total_energy_mj += avg_power * dt
                
                self._last_time = now
                self._last_power = power_mw
                time.sleep(0.005)
            except Exception:
                break

    def shutdown(self):
        pynvml.nvmlShutdown()

class GPU16bitEmbedding(nn.Module):
    def __init__(self, source_layer):
        super().__init__()
        self.num_embeddings = source_layer.num_embeddings
        self.embedding_dim = source_layer.embedding_dim
        self.weight = nn.Parameter(source_layer.weight.data.detach().cuda().half())

    def forward(self, x):
        return F.embedding(x, self.weight).to(torch.float32)

class BNB8bitEmbedding(nn.Module):
    def __init__(self, source_layer, chunk_size=2048):
        super().__init__()
        self.num_embeddings = source_layer.num_embeddings
        self.embedding_dim = source_layer.embedding_dim
        self.chunk_size = chunk_size or self.num_embeddings

        w = source_layer.weight.data.detach().cuda()

        for chunk_idx, i in enumerate(range(0, self.num_embeddings, self.chunk_size)):
            end_i = min(i + self.chunk_size, self.num_embeddings)
            chunk_w = w[i:end_i]  # (N, D)

            absmax = chunk_w.abs().amax(dim=1, keepdim=True) + 1e-8

            normed = chunk_w / absmax

            q_weight = torch.clamp((normed * 127).round(), -128, 127).to(torch.int8)

            self.register_buffer(f"q_weight_{chunk_idx}", q_weight)
            self.register_buffer(f"absmax_{chunk_idx}", absmax.squeeze(1))

    def forward(self, x):
        if (x >= self.num_embeddings).any():
            raise IndexError(f"ID {x.max().item()} out of bounds")

        out = torch.empty((*x.shape, self.embedding_dim),
                          dtype=torch.float32,
                          device=x.device)

        for chunk_idx, i in enumerate(range(0, self.num_embeddings, self.chunk_size)):
            end_i = min(i + self.chunk_size, self.num_embeddings)

            mask = (x >= i) & (x < end_i)
            if not mask.any():
                continue

            q_w = getattr(self, f"q_weight_{chunk_idx}")
            absmax = getattr(self, f"absmax_{chunk_idx}")

            local_indices = x[mask] - i

            selected_qw = q_w[local_indices].float()
            selected_absmax = absmax[local_indices].unsqueeze(1)

            out[mask] = selected_qw * (selected_absmax / 127.0)

        return out

class BNB4bitEmbedding(nn.Module):
    def __init__(self, source_layer, chunk_size=2048, quant_type="nf4"):
        super().__init__()
        self.num_embeddings = source_layer.num_embeddings
        self.embedding_dim = source_layer.embedding_dim
        self.chunk_size = chunk_size or self.num_embeddings
        self.quant_type = quant_type.lower()

        assert self.quant_type in ["fp4", "nf4"]

        device = source_layer.weight.device
        w = source_layer.weight.data.detach().to(device)

        if self.quant_type == "nf4":
            self.register_buffer(
                "codebook",
                torch.tensor([
                    -1.0000, -0.6962, -0.5251, -0.3949,
                    -0.2844, -0.1848, -0.0911,  0.0000,
                     0.0796,  0.1609,  0.2461,  0.3379,
                     0.4407,  0.5626,  0.7230,  1.0000
                ], dtype=torch.float32, device=device)
            )

        for chunk_idx, i in enumerate(range(0, self.num_embeddings, self.chunk_size)):
            end_i = min(i + self.chunk_size, self.num_embeddings)
            chunk_w = w[i:end_i]  # (N, D)

            absmax = chunk_w.abs().amax(dim=1, keepdim=True) + 1e-8
            normed = chunk_w / absmax  # [-1,1]

            if self.quant_type == "fp4":
                q = torch.clamp(((normed + 1) * 7.5).round(), 0, 15)

            else:  # nf4
                cb = self.codebook.view(1, 1, -1)
                dist = torch.abs(normed.unsqueeze(-1) - cb)
                q = dist.argmin(dim=-1)

            q = q.to(torch.uint8)  # (N, D)

            if self.embedding_dim % 2 != 0:
                q = torch.cat([q, torch.zeros(q.size(0), 1, device=q.device, dtype=q.dtype)], dim=1)

            q_even = q[:, 0::2]
            q_odd  = q[:, 1::2]

            packed = (q_even << 4) | q_odd  # (N, D/2)

            self.register_buffer(f"q_weight_{chunk_idx}", packed)
            self.register_buffer(f"absmax_{chunk_idx}", absmax.squeeze(1))

    def forward(self, x):
        if (x >= self.num_embeddings).any():
            raise IndexError(f"ID {x.max().item()} out of bounds")

        out = torch.empty((*x.shape, self.embedding_dim),
                          dtype=torch.float32,
                          device=x.device)

        for chunk_idx, i in enumerate(range(0, self.num_embeddings, self.chunk_size)):
            end_i = min(i + self.chunk_size, self.num_embeddings)

            mask = (x >= i) & (x < end_i)
            if not mask.any():
                continue

            packed = getattr(self, f"q_weight_{chunk_idx}")
            absmax = getattr(self, f"absmax_{chunk_idx}")

            local_indices = x[mask] - i

            selected = packed[local_indices]  # (M, D/2)

            high = (selected >> 4) & 0xF
            low  = selected & 0xF

            q = torch.empty(
                (selected.shape[0], selected.shape[1] * 2),
                dtype=torch.uint8,
                device=selected.device
            )

            q[:, 0::2] = high
            q[:, 1::2] = low

            q = q[:, :self.embedding_dim]

            if self.quant_type == "fp4":
                normed = (q.float() / 7.5) - 1.0
            else:
                normed = self.codebook[q.long()]

            out[mask] = normed * absmax[local_indices].unsqueeze(1)

        return out

def BNBFP4Embedding(layer): return BNB4bitEmbedding(layer, quant_type="fp4")
def BNBNF4Embedding(layer): return BNB4bitEmbedding(layer, quant_type="nf4")

class TiedEmbeddingLinear(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.in_features = embedding_layer.embedding_dim
        self.out_features = embedding_layer.num_embeddings

    def forward(self, x):
        if hasattr(self.embedding_layer, "weight"):
            return F.linear(x, self.embedding_layer.weight.to(torch.float32))

        logits = torch.empty(
            (*x.shape[:-1], self.out_features),
            dtype=torch.float32,
            device=x.device
        )

        chunk_size = self.embedding_layer.chunk_size
        quant_type = getattr(self.embedding_layer, "quant_type", None)

        for chunk_idx, i in enumerate(range(0, self.out_features, chunk_size)):
            end_i = min(i + chunk_size, self.out_features)

            q_w = getattr(self.embedding_layer, f"q_weight_{chunk_idx}")
            absmax = getattr(self.embedding_layer, f"absmax_{chunk_idx}")

            if q_w.dtype == torch.int8:
                chunk_w_fp = q_w.float() * (absmax[:, None] / 127.0)

            elif q_w.dtype == torch.uint8:
                high = (q_w >> 4) & 0xF
                low  = q_w & 0xF

                q = torch.empty(
                    (q_w.shape[0], q_w.shape[1] * 2),
                    dtype=torch.uint8,
                    device=q_w.device
                )

                q[:, 0::2] = high
                q[:, 1::2] = low
                q = q[:, :self.in_features]

                if quant_type == "nf4":
                    codebook = self.embedding_layer.codebook
                    normed = codebook[q.long()]
                else:
                    normed = (q.float() / 7.5) - 1.0

                chunk_w_fp = normed * absmax[:, None]

            else:
                raise TypeError("Unsupported quantization type")

            logits[..., i:end_i] = F.linear(x, chunk_w_fp)

            del chunk_w_fp

        return logits

def set_seed(seed=None):
    if seed is None:
        return

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


