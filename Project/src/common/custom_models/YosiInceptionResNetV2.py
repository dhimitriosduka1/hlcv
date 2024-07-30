import urllib
from PIL import Image

import timm
import torch.nn as nn

class YosiInceptionResNetV2(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(YosiInceptionResNetV2, self).__init__()
        
        # Load pre-trained InceptionResNetV2 from timm
        self.base_model = timm.create_model('inception_resnet_v2', pretrained=pretrained, num_classes=num_classes)
        
        self.base_model.global_pool = nn.Identity()
        self.base_model.classif = nn.Identity()

        self.layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten(),
            nn.Linear(38400, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        ])
        
    def forward(self, x):
        x = self.base_model(x)
        for layer in self.layers:
            x = layer(x)
        return x

model = YosiInceptionResNetV2(num_classes=7)

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)

img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0)

import torch
with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)
# prints: torch.Size([1000])