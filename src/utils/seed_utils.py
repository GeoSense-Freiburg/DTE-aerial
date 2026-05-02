import os
import random
import numpy as np
import torch

def set_random_seed(seed, rank=0):
    """
    Set the random seed for reproducibility across various libraries.
    
    Args:
        seed (int): The base seed value.
        rank (int): The rank of the current process in the distributed setup.
    """
    # Calculate a unique seed for this process
    unique_seed = seed + rank
    # Set seed for Python's built-in random module
    random.seed(unique_seed)
    # Set seed for NumPy
    np.random.seed(unique_seed)
    # Set seed for PyTorch
    torch.manual_seed(unique_seed)
    torch.cuda.manual_seed_all(unique_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #Enable only if full reproducibility is required.
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True