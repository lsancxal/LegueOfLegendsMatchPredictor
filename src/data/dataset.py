"""Custom PyTorch Dataset for match data."""

import torch
from torch.utils.data import Dataset


class MatchDataset(Dataset):
    """Dataset for League of Legends match data."""
    
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            features: Input features tensor
            labels: Target labels tensor
        """
        self.features = features
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
