import torch
import torch.nn as nn
from torchvision import models


class EyeDiseaseClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(EyeDiseaseClassifier, self).__init__()

        # Load ResNet18
        self.model = models.resnet18(pretrained=pretrained)

        # Freeze feature extractor (optional but recommended initially)
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
