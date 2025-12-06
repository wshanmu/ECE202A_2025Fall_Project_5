import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock2D(nn.Module):
    """
    A basic ResNet block for 2D inputs.
    Maintains spatial dimensions (Time, Antennas) to allow residual addition.
    """
    def __init__(self, channels, kernel_size=(3, 3)):
        super(ResNetBlock2D, self).__init__()
        
        # Calculate padding to maintain tensor shape
        # padding = k // 2 assumes odd kernel size and stride 1
        pad_time = kernel_size[0] // 2
        pad_ant = kernel_size[1] // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                               stride=1, padding=(pad_time, pad_ant))
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                               stride=1, padding=(pad_time, pad_ant))
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        out += residual
        out = F.relu(out)
        return out

class ResNetBlock1D(nn.Module):
    """
    Standard 1D ResNet block for temporal processing.
    """
    def __init__(self, channels, kernel_size=3):
        super(ResNetBlock1D, self).__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AntennaHybridNet(nn.Module):
    def __init__(self, input_channels=1, base_channels=64, num_classes=1):
        super(AntennaHybridNet, self).__init__()
        
        # --- Stage 1: Antenna-Specific Feature Extraction (2D) ---
        # Input: (BS, 1, T, 6)
        # Kernel (7, 2) with stride (2, 2)
        # Time dimension reduces by factor of 2.
        # Feature dimension (6) reduces to 3 (representing 3 antennas).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=base_channels, 
                      kernel_size=(10, 2), stride=(2, 2), padding=(3, 0)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        # --- Stage 2: 2D Residual Processing ---
        # Input: (BS, 64, T/2, 3)
        # We use a kernel of (3, 3) with padding (1, 1) to maintain the "3" width.
        # This mixes information between antennas while maintaining the structure.
        self.res_block_2d = ResNetBlock2D(base_channels, kernel_size=(3, 3))
        
        # --- Stage 3: Aggregation ---
        # We pool across the feature dimension (width) only.
        # This fuses the 3 antenna streams into one timeline.
        # We perform this manually in forward() using mean().
        
        # --- Stage 4: Temporal Processing (1D) ---
        # Input: (BS, 64, T/2)
        self.res_block_1d_1 = ResNetBlock1D(base_channels)
        self.res_block_1d_2 = ResNetBlock1D(base_channels)
        
        # --- Stage 5: Classification ---
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.ReLU(),
            nn.Linear(base_channels, num_classes),
        )
    def forward(self, x):
        # x shape: (BS, 1, T, 6)
        
        # 1. Stem
        x = self.stem(x)
        # x shape: (BS, 64, T/2, 3)
        
        # 2. 2D ResNet
        x = self.res_block_2d(x)
        # x shape: (BS, 64, T/2, 3)
        
        # 3. Aggregation (Pool over the 'antenna' dimension, which is dim 3)
        # We want to keep (BS, Channel, Time) -> (BS, 64, T/2)
        x = x.mean(dim=3) 
        
        # 4. 1D ResNet
        x = self.res_block_1d_1(x)
        x = self.res_block_1d_2(x)
        # x shape: (BS, 64, T/2)
        
        # 5. Classifier Head
        x = self.global_pool(x)  # (BS, 64, 1)
        x = x.flatten(1)         # (BS, 64)
        x = self.classifier(x)   # (BS, 1)
        
        return x

# --- Testing the Block ---
if __name__ == "__main__":
    # Dummy Input: Batch=8, Channel=1, Time=1024, Features=6
    # Features=6 corresponds to (Amp1, Ph1, Amp2, Ph2, Amp3, Ph3)
    bs, t = 8, 1536
    dummy_input = torch.randn(bs, 1, t, 48)
    
    model = AntennaHybridNet(num_classes=2)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check intermediate shapes by hooking or manual inspection
    stem_out = model.stem(dummy_input)
    print(f"Shape after Stem (expecting T/2 and width 3): {stem_out.shape}")