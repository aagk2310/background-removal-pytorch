import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(SqueezeAndExcite, self).__init__()
        self.in_channels = in_channels
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        se = self.se(x)
        return x * se

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]

        y1 = F.interpolate(self.conv1(x), size=size, mode='bilinear', align_corners=True)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y4 = self.conv4(x)
        y5 = self.conv5(x)

        y = torch.cat([y1, y2, y3, y4, y5], dim=1)
        y = self.output_conv(y)
        return y

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLabV3Plus, self).__init__()

        # Load pre-trained ResNet50 model
        self.backbone = models.resnet50(pretrained=True)
        self.backbone_layers = list(self.backbone.children())

        # Use the conv2_block2 and conv4_block6 equivalent outputs
        self.conv2_block2_out = nn.Sequential(*self.backbone_layers[:5])  # Until conv2 block
        self.conv4_block6_out = nn.Sequential(*self.backbone_layers[5:7])  # Until conv4 block

        self.aspp = ASPP(1024, 256)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.se1 = SqueezeAndExcite(304)  # After concatenation of aspp and shortcut
        self.final_conv1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.final_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.se2 = SqueezeAndExcite(256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]

        # Encoder
        conv2 = self.conv2_block2_out(x)
        conv4 = self.conv4_block6_out(conv2)

        # ASPP
        x_a = self.aspp(conv4)
        x_a = F.interpolate(x_a, scale_factor=4, mode='bilinear', align_corners=True)

        # Shortcut
        x_b = self.shortcut_conv(conv2)

        # Concatenate ASPP and shortcut, and apply Squeeze and Excitation
        x = torch.cat([x_a, x_b], dim=1)
        x = self.se1(x)

        # Final convolutions
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        x = self.se2(x)

        # Upsampling and classification
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.classifier(x)
        x = torch.sigmoid(x)

        return x
