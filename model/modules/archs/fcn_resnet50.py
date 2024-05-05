import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50

class FCNResNet(nn.Module):
    name = 'fcn_resnet50'

    def __init__(self, options):
        super().__init__()
        self.out_channels = options.get('out_channels')
        self.model = fcn_resnet50(num_classes=self.out_channels)

    def forward(self, x):
        return self.model(x)
