import torch
import torch.nn as nn

config = {
    'VGG-16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGGNet(nn.Module):
    def __init__(self, vgg_model):
        super(VGGNet, self).__init__()

        self.features = self._make_layer(config[vgg_model])
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def _make_layer(self, config):
        layers = []
        in_channels = 3

        for a in config:
            if a == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, a, kernel_size=3, padding=1),
                           nn.BatchNorm2d(a),
                           nn.ReLU(inplace=True)]
                in_channels = a
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def VGG16():
    return VGGNet('VGG-16')