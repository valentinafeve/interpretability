from torch import nn
from training.icnn import InterpretableConvLayer

class Model(nn.Module):
    def __init__(self, feature_map_size, channels=1, num_classes=10, filters_icnn=6):
        super().__init__()
        filters_cnn=20
        self.cnn = nn.Conv2d(channels, filters_cnn, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn2 = nn.Conv2d(in_channels=filters_cnn, out_channels=filters_icnn, kernel_size=3, padding=1)

        input_size = filters_icnn * (feature_map_size // 4) * (feature_map_size // 4)
        self.linear = nn.Linear(3072, 100)
        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x)
        x = nn.ReLU()(x)
        x = self.cnn2(x)
        x = nn.ReLU()(x)
        filters = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = nn.ReLU()(x)
        logits = self.classifier(x)
        return logits, filters
