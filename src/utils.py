import random
import numpy as np
import torch
import os

def set_seed(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_seed(seed=None):
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Set standard random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    
    if seed is not None:
        # 3. Force completely deterministic CUDA algorithms
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # Required for PyTorch 1.8+
        torch.use_deterministic_algorithms(True)
        
        # 4. Configure CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False