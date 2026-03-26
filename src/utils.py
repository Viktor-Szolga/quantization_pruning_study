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
    def __init__(self, source_layer, chunk_size=64):
        super().__init__()
        self.num_embeddings = source_layer.num_embeddings
        self.embedding_dim = source_layer.embedding_dim
        if chunk_size is None:
            self.chunk_size = self.num_embeddings
        else:
            self.chunk_size = chunk_size
        
        w = source_layer.weight.data.detach().cuda()
        self.q_states = []
        
        for chunk_idx, i in enumerate(range(0, self.num_embeddings, self.chunk_size)):
            end_i = min(i + self.chunk_size, self.num_embeddings)
            chunk_w = w[i:end_i]
            
            q_weight, q_state = bnb.functional.quantize_blockwise(chunk_w)
            
            self.register_buffer(f"q_weight_{chunk_idx}", q_weight)
            if hasattr(q_state, 'absmax'):
                self.register_buffer(f"absmax_{chunk_idx}", q_state.absmax)
            if hasattr(q_state, 'code'):
                self.register_buffer(f"code_{chunk_idx}", q_state.code)
                
            self.q_states.append(q_state)

    def forward(self, x):
        if (x >= self.num_embeddings).any():
            max_id = x.max().item()
            raise IndexError(f"ID {max_id} is out of bounds for embedding size {self.num_embeddings}")


        out = torch.empty((*x.shape, self.embedding_dim), dtype=torch.float32, device=x.device)
        
        for chunk_idx, i in enumerate(range(0, self.num_embeddings, self.chunk_size)):
            end_i = min(i + self.chunk_size, self.num_embeddings)
            
            mask = (x >= i) & (x < end_i)
            
            if mask.any():
                q_w = getattr(self, f"q_weight_{chunk_idx}")
                q_s = self.q_states[chunk_idx]
                
                chunk_w_fp = bnb.functional.dequantize_blockwise(q_w, q_s)
                
                local_indices = x[mask] - i
                out[mask] = chunk_w_fp[local_indices]
                
                del chunk_w_fp 
                
        return out


class BNB4bitEmbedding(nn.Module):
    def __init__(self, source_layer, chunk_size=64, quant_type="nf4"):
        super().__init__()
        self.num_embeddings = source_layer.num_embeddings
        self.embedding_dim = source_layer.embedding_dim
        if chunk_size is None:
            self.chunk_size = self.num_embeddings
        else:
            self.chunk_size = chunk_size
        
        w = source_layer.weight.data.detach().cuda()
        self.q_states = []
        
        for chunk_idx, i in enumerate(range(0, self.num_embeddings, self.chunk_size)):
            end_i = min(i + self.chunk_size, self.num_embeddings)
            chunk_w = w[i:end_i]
            
            q_weight, q_state = bnb.functional.quantize_4bit(chunk_w, quant_type=quant_type)
            
            self.register_buffer(f"q_weight_{chunk_idx}", q_weight)
            if hasattr(q_state, 'absmax'):
                self.register_buffer(f"absmax_{chunk_idx}", q_state.absmax)
                
            self.q_states.append(q_state)

    def forward(self, x):
        if (x >= self.num_embeddings).any():
            max_id = x.max().item()
            raise IndexError(f"ID {max_id} is out of bounds for embedding size {self.num_embeddings}")

        out = torch.empty((*x.shape, self.embedding_dim), dtype=torch.float32, device=x.device)
        
        for chunk_idx, i in enumerate(range(0, self.num_embeddings, self.chunk_size)):
            end_i = min(i + self.chunk_size, self.num_embeddings)
            
            mask = (x >= i) & (x < end_i)
            
            if mask.any():
                q_w = getattr(self, f"q_weight_{chunk_idx}")
                q_s = self.q_states[chunk_idx]
                
                chunk_w_fp = bnb.functional.dequantize_4bit(q_w, q_s)
                
                local_indices = x[mask] - i
                out[mask] = chunk_w_fp[local_indices]
                
                del chunk_w_fp 
                
        return out

def BNBFP4Embedding(layer): return BNB4bitEmbedding(layer, quant_type="fp4")
def BNBNF4Embedding(layer): return BNB4bitEmbedding(layer, quant_type="nf4")

class TiedEmbeddingLinear(nn.Module):
    def __init__(self, embedding_layer):
        """
        A weight-tied output layer (bias=False) compatible with chunked 
        16-bit, 8-bit, and 4-bit embedding classes.
        """
        super().__init__()
        self.embedding_layer = embedding_layer
        self.in_features = embedding_layer.embedding_dim
        self.out_features = embedding_layer.num_embeddings

    def forward(self, x):
        """
        x: [batch, seq_len, hidden_dim]
        returns: [batch, seq_len, vocab_size]
        """
        if hasattr(self.embedding_layer, 'weight'):
            return F.linear(x, self.embedding_layer.weight.to(torch.float32))

        elif hasattr(self.embedding_layer, 'q_states'):
            logits = torch.empty(
                (*x.shape[:-1], self.out_features), 
                dtype=torch.float32, 
                device=x.device
            )
            
            chunk_size = self.embedding_layer.chunk_size
            is_4bit = "4bit" in type(self.embedding_layer).__name__.lower()
            
            for chunk_idx, i in enumerate(range(0, self.out_features, chunk_size)):
                end_i = min(i + chunk_size, self.out_features)
                
                q_w = getattr(self.embedding_layer, f"q_weight_{chunk_idx}")
                q_s = self.embedding_layer.q_states[chunk_idx]
                
                if is_4bit:
                    chunk_w_fp = bnb.functional.dequantize_4bit(q_w, q_s)
                else:
                    chunk_w_fp = bnb.functional.dequantize_blockwise(q_w, q_s)
                
                logits[..., i:end_i] = F.linear(x, chunk_w_fp)
                
                del chunk_w_fp 
                
            return logits
            
        else:
            raise TypeError("The provided embedding layer is not a recognized type.")


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


