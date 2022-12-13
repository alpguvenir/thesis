import torch
import torch.nn as nn
from torchvision import models

class AttentionClassifier(nn.Module):
    

    def __init__(self):
        super().__init__()

        get_params = lambda m: sum(p.numel() for p in m.parameters())

        hidden_size1 = 256
        feature_extractor = models.resnet.resnet18(pretrained=True)

        feature_extractor.fc = (
            nn.Linear(512, hidden_size1) if hidden_size1 != 512 else nn.Identity()
        )
        feature_extractor.conv1 = nn.Sequential(
            nn.Conv2d(1, 3, 1), feature_extractor.conv1
        )
        
        self.feature_extractor = feature_extractor
        self.att = nn.MultiheadAttention(hidden_size1, 8)
        self.classifier = nn.Linear(hidden_size1, 1)

        print(f"Feature extractor has {get_params(self.feature_extractor)} params")
        print(f"Attention has {get_params(self.att)} params")
        print(f"Classifier has {get_params(self.classifier)} params")

    def forward(self, x):
        features = self.feature_extractor(x).unsqueeze(
            1
        )  # assuming only 1 CT at a time
        query = features.mean(0, keepdims=True)
        # print(features.shape)
        # print(query.shape)
        features, att_map = self.att(query, features, features)
        out = self.classifier(features.squeeze(0))
        # print(out.shape)
        return out