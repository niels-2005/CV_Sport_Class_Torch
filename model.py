import timm
from torch import nn


class HotdogClassifier(nn.Module):
    def __init__(self, num_classes=15, dropout_rate=0.50):
        super(HotdogClassifier, self).__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(enet_out_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
