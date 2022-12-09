import json
import os

import torch
import numpy as np

import typing
from typing import Any, Type

class Dataset(torch.utils.data.Dataset):

    def __init__(self, scans: list[str], labels: list[str], transform: dict = None, scan_type: Type[Any] = np.float32, label_type: Type[Any] = np.int64):

        self.scans = scans
        self.labels = labels

        self.transform = transform
        self.scan_type = scan_type
        self.label_type = label_type


    def __len__(self) -> int:
        return len(self.scans)

    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        scan, label = (
            self.scans[index],
            self.labels[index]   
        )

        """
        # Apply transformation to scan image
        if self.transform:
            self.transform(scan, index)
        """
        return scan, label

