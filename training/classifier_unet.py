import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()
        self.pooling = pooling
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if self.pooling:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, depth=5, start_filts=64):
        super(UNetEncoder, self).__init__()
        self.depth = depth
        self.start_filts = start_filts
        self.down_convs = nn.ModuleList()
        self.out_channels = start_filts * (2 ** (depth - 1))

        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filts * (2 ** i)
            pooling = i < depth - 1
            self.down_convs.append(DownConv(ins, outs, pooling=pooling))
    
    def forward(self, x):
        for module in self.down_convs:
            x, _ = module(x)
        return x  # Final feature map

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten feature map
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ClassifierUNet(nn.Module):
    def __init__(self, in_channels, img_size, depth=5, start_filts=64, hidden_dim=128, output_dim=1):
        super(ClassifierUNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, depth, start_filts)
        final_feature_map_size = img_size // (2 ** (depth - 1))
        input_dim = self.encoder.out_channels * final_feature_map_size * final_feature_map_size
        self.mlp = MLP(input_dim, hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        return x
