import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchvision import models

class ImageBinaryModel(nn.Module):
    """
    Model for binary image classification
    """
    def __init__(self, base_model="vgg16", freeze_base=False):
        super().__init__()
        
        if base_model=="vgg16":
            self.base = models.vgg16(pretrained=True)
        else:
            raise ValueError("Invalid base model!")
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )
        self.m = nn.Sigmoid()

        if freeze_base:
            for p in [
                *self.base.parameters(),
            ]:
                p.requires_grad = False
    
    def forward(self, data):
        base_output = self.base(data)
        x = self.relu(base_output)
        x = self.dropout(x)
        output = self.classifier(x)
        return self.m(output)


if __name__ == "__main__":
    model = ImageBinaryModel()
    # x, y
    output = model(x)

