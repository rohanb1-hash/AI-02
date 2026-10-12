import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Improve reproducibility for cuDNN operations.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
