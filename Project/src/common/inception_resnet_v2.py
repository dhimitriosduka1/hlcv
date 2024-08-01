import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import transforms

class InceptionResnetV2(nn.Module):
    def __init__(self, num_classes=7):
        super(InceptionResnetV2, self).__init__()
        
        # Load pre-trained InceptionResNetV2 from timm
        self.base_model = timm.create_model('inception_resnet_v2', pretrained=True)
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Remove the original classification layer
        num_ftrs = self.base_model.classif.in_features
        self.base_model.classif = nn.Identity()
        
        # Define the sequential model for additional layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten(),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x
