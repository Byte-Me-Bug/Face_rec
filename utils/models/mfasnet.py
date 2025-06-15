import torch.nn as nn
import torch.nn.functional as F


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1):
        super(DepthWiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(nin, nin, kernel_size, stride, padding, groups=nin, bias=False),
            nn.BatchNorm2d(nin),
            nn.ReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MiniFASNetV1SE(nn.Module):
    def __init__(self, num_classes=2, input_size=(80, 80)):
        super(MiniFASNetV1SE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.conv2_dw = DepthWiseSeparableConv(16, 32)
        self.conv3_dw = DepthWiseSeparableConv(32, 64)
        self.conv4_dw = DepthWiseSeparableConv(64, 128)

        final_size = input_size[0] // 8  # After 3 strides
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv3_dw(x)
        x = self.conv4_dw(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
