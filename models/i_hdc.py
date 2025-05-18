import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedReceptiveFieldModule(nn.Module):
    def __init__(self, in_channels, out_channels, stochastic_depth_prob=0.2):
        super(MixedReceptiveFieldModule, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=2, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=3, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.gelu = nn.GELU()
        self.stochastic_depth_prob = stochastic_depth_prob
    
    def forward(self, x):
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x = torch.cat([x3, x5], dim=1)
        x = self.bn(x)
        x = self.gelu(x)
        
        # Stochastic depth
        if self.training and torch.rand(1).item() < self.stochastic_depth_prob:
            return x * 0
        return x

class AdaptiveFeatureExtractionModule(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(AdaptiveFeatureExtractionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.gelu1 = nn.GELU()
        self.pool1 = nn.AdaptiveMaxPool2d(16)
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu2 = nn.GELU()
        self.pool2 = nn.AdaptiveAvgPool2d(8)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu2(x)
        x = self.pool2(x)
        return x

class iHDC(nn.Module):
    def __init__(self, num_classes, in_channels=3, base_channels=32, dropout_rate=0.5, stochastic_depth_prob=0.2):
        super(iHDC, self).__init__()
        self.mrf = MixedReceptiveFieldModule(in_channels, base_channels, stochastic_depth_prob)
        self.feature_extraction = AdaptiveFeatureExtractionModule(base_channels * 2, 64, 128)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 8 * 8, 512)
        self.ln = nn.LayerNorm(512)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.mrf(x)
        x = self.feature_extraction(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def i_hdc(num_classes, **kwargs):
    return iHDC(num_classes, **kwargs)
