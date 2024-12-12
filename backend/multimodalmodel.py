import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock

class ResNet10(ResNet):
    def __init__(self, num_classes=1000):
        super(ResNet10, self).__init__(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
        self.fc = nn.Linear(512, num_classes)

class MultimodalModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes):
        super(MultimodalModel, self).__init__()

        # image branch with ResNet
        self.cnn = ResNet10(num_classes=512)
        self.cnn.fc = nn.Identity()

        # tabular branch
        self.tabular = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64 + 512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, tabular_features):
        # image features
        image_features = self.cnn(image)

        # tabular features
        tabular_features = self.tabular(tabular_features)

        # combine features
        combined = torch.cat((image_features, tabular_features), dim=1)

        # final classification
        output = self.fusion(combined)
        return output
