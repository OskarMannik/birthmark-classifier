import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18

class MultimodalModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes):
        super(MultimodalModel, self).__init__()

        # image branch with ResNet
        self.cnn = resnet18(pretrained=False)
        self.cnn.fc = nn.Identity()

        # tabular branch
        self.tabular = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64 + 512, 128),
            nn.ReLU(),
            #nn.Dropout(p=0.5),
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
