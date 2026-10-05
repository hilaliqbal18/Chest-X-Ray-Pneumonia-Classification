import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model