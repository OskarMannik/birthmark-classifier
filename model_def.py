import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes):
        super(MultimodalModel, self).__init__()

        # Image branch (pre-trained ResNet)
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the ResNet final classification layer

        # Tabular branch
        self.tabular = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64 + 512, 128),  # 512 from ResNet output, 64 from tabular branch
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, tabular_features):
        # Image features
        image_features = self.cnn(image)

        # Tabular features
        tabular_features = self.tabular(tabular_features)

        # Combine features
        combined = torch.cat((image_features, tabular_features), dim=1)

        # Final classification
        output = self.fusion(combined)
        return output
