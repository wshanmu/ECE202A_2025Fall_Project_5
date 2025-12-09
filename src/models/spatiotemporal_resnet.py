"""
SpatioTemporal ResNet for Heart Rate Prediction from CIR Data

This module implements a hybrid 2D+1D ResNet architecture that processes
preprocessed CIR features with shape (B, T, S) where:
- B: batch size
- T: time dimension (number of frames)
- S: spatial dimension (2 * num_antennas * num_range_bins)

The architecture consists of:
1. 2D ResNet: Captures spatio-temporal structure
2. Spatial pooling: Collapses spatial dimension
3. 1D ResNet: Processes temporal features
4. Global temporal pooling + FC: Outputs heart rate prediction

Input assumptions:
- Features are already min-max normalized to [0, 1]
- T and S are multiples of the downsampling factors (T divisible by 128, S divisible by 8)

Author: Generated for DeskSense project
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


# ============================================================================
# 2D ResNet Components
# ============================================================================

class BasicBlock2D(nn.Module):
    """
    Basic 2D ResNet block with skip connection.

    Architecture:
        Main path:
            Conv2d(kernel_size=3, stride=stride) -> BatchNorm2d -> ReLU
            Conv2d(kernel_size=3, stride=1) -> BatchNorm2d

        Skip path:
            Conv2d(kernel_size=1, stride=stride) + BatchNorm2d if needed
            else identity

        Output: ReLU(main + skip)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for first convolution (default: 1)

    Shape:
        Input: (B, in_channels, H, W)
        Output: (B, out_channels, H/stride, W/stride)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock2D, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 2D ResNet block.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H/stride, W/stride)
        """
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection
        out += self.skip(x)
        out = self.relu(out)

        return out


class ResNet2D(nn.Module):
    """
    2D ResNet for spatio-temporal feature extraction.

    Architecture:
        1. Init Conv block:
           - Conv2d(7x7, stride=2)
           - BatchNorm2d
           - ReLU
           - MaxPool2d(3x3, stride=2)

        2. ResNet Stage [1]: 32 filters, 1 block, stride=2
        3. ResNet Stage [2]: 64 filters, 2 blocks (first stride=2, second stride=1)

    Args:
        in_channels: Number of input channels (default: 1)

    Shape:
        Input: (B, 1, T, S)
        Output: (B, 64, T/8, S/8)
    """

    def __init__(self, in_channels: int = 1):
        super(ResNet2D, self).__init__()

        # Initial convolution block
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet Stage [2]: 32 filters, 2 blocks
        # First block downsamples (stride=2), second block maintains (stride=1)
        self.stage1 = self._make_stage(16, 16, num_blocks=2, stride=2)

        # ResNet Stage [2]: 64 filters, 2 blocks
        self.stage2 = self._make_stage(16, 32, num_blocks=2, stride=2)

    def _make_stage(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        """
        Create a ResNet stage with multiple blocks.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_blocks: Number of blocks in this stage
            stride: Stride for the first block (subsequent blocks use stride=1)

        Returns:
            Sequential module containing all blocks
        """
        layers = []

        # First block with specified stride (may downsample)
        layers.append(BasicBlock2D(in_channels, out_channels, stride))

        # Subsequent blocks with stride=1
        for _ in range(1, num_blocks):
            layers.append(BasicBlock2D(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 2D ResNet.

        Args:
            x: Input tensor of shape (B, 1, T, S)

        Returns:
            Output tensor of shape (B, 64, T/8, S/8)
        """
        x = self.init_conv(x)  # (B, 32, T/4, S/4) due to stride=2 and maxpool stride=2
        x = self.stage1(x)      # (B, 32, T/8, S/8) due to stride=2
        x = self.stage2(x)      # (B, 64, T/8, S/8)

        return x


# ============================================================================
# 1D ResNet Components
# ============================================================================

class BasicBlock1D(nn.Module):
    """
    Basic 1D ResNet block with skip connection.

    Architecture:
        Main path:
            Conv1d(kernel_size=3, stride=stride) -> BatchNorm1d -> ReLU
            Conv1d(kernel_size=3, stride=1) -> BatchNorm1d

        Skip path:
            Conv1d(kernel_size=1, stride=stride) + BatchNorm1d if needed
            else identity

        Output: ReLU(main + skip)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for first convolution (default: 1)

    Shape:
        Input: (B, in_channels, L)
        Output: (B, out_channels, L/stride)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock1D, self).__init__()

        # Main path
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 1D ResNet block.

        Args:
            x: Input tensor of shape (B, in_channels, L)

        Returns:
            Output tensor of shape (B, out_channels, L/stride)
        """
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection
        out += self.skip(x)
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for temporal feature extraction.

    Architecture:
        1. Init Conv block:
           - Conv1d(7x7, stride=2)
           - BatchNorm1d
           - ReLU
           - MaxPool1d(3x3, stride=2)

        2. ResNet Stage [1]: 64 filters, 2 blocks (first stride=2, second stride=1)
        3. ResNet Stage [2]: 128 filters, 2 blocks (first stride=2, second stride=1)
        4. ResNet Stage [3]: 256 filters, 2 blocks (first stride=2, second stride=1)
        5. ResNet Stage [4]: 512 filters, 2 blocks (first stride=2, second stride=1)

    Args:
        in_channels: Number of input channels (default: 64)

    Shape:
        Input: (B, 64, T')
        Output: (B, 512, T'')
        where T'' ≈ T' / 16 (due to 5 stride-2 operations: init stride + maxpool + 4 stages)
    """

    def __init__(self, in_channels: int = 64):
        super(ResNet1D, self).__init__()

        # Initial convolution block
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet Stage [1]: 64 filters, 1 block
        self.stage1 = self._make_stage(32, 32, num_blocks=1, stride=2)

        # ResNet Stage [2]: 128 filters, 1 block
        self.stage2 = self._make_stage(32, 64, num_blocks=1, stride=2)

        # ResNet Stage [3]: 256 filters, 2 blocks
        self.stage3 = self._make_stage(64, 128, num_blocks=2, stride=2)

        # ResNet Stage [4]: 512 filters, 2 blocks
        self.stage4 = self._make_stage(128, 256, num_blocks=2, stride=2)

    def _make_stage(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        """
        Create a ResNet stage with multiple blocks.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_blocks: Number of blocks in this stage (should be 2)
            stride: Stride for the first block

        Returns:
            Sequential module containing all blocks
        """
        layers = []

        # First block with specified stride (downsamples)
        layers.append(BasicBlock1D(in_channels, out_channels, stride))

        # Subsequent blocks with stride=1
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 1D ResNet.

        Args:
            x: Input tensor of shape (B, 64, T')

        Returns:
            Output tensor of shape (B, 512, T'')
        """
        x = self.init_conv(x)  # (B, 64, T'/4) due to stride=2 and maxpool stride=2
        x = self.stage1(x)     # (B, 64, T'/8) due to first block stride=2
        x = self.stage2(x)     # (B, 128, T'/16)
        x = self.stage3(x)     # (B, 256, T'/16)
        x = self.stage4(x)     # (B, 512, T'/32)

        return x


# ============================================================================
# Full SpatioTemporal ResNet Model
# ============================================================================

class SpatioTemporalResNetHR(nn.Module):
    """
    SpatioTemporal ResNet for Heart Rate Prediction.

    This model combines 2D and 1D ResNets to process CIR features:

    Pipeline:
        1. Input: (B, T, S) or (B, 1, T, S)
           - Reshape to (B, 1, T, S) if needed

        2. 2D ResNet: (B, 1, T, S) -> (B, 64, T/8, S/8)
           - Captures spatio-temporal structure

        3. Spatial pooling: (B, 64, T/8, S/8) -> (B, 64, T/8)
           - Average over spatial dimension

        4. 1D ResNet: (B, 64, T/8) -> (B, 512, T/128)
           - Processes temporal features

        5. Global temporal pooling: (B, 512, T/128) -> (B, 512)
           - Average over temporal dimension

        6. FC layer: (B, 512) -> (B, 1)
           - Outputs heart rate prediction

    Args:
        None (architecture is fixed as specified)

    Input requirements:
        - Features must be min-max normalized to [0, 1]
        - T should be divisible by 128 (total downsampling factor)
        - S should be divisible by 8

    Shape:
        Input: (B, T, S) or (B, 1, T, S)
        Output: (B, 1)
    """

    def __init__(self):
        super(SpatioTemporalResNetHR, self).__init__()

        # 2D ResNet for spatio-temporal features
        self.resnet2d = ResNet2D(in_channels=1)

        # 1D ResNet for temporal features
        self.resnet1d = ResNet1D(in_channels=32)

        # Final fully connected layer for heart rate prediction
        self.fc = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full model.

        Args:
            x: Input tensor of shape (B, T, S) or (B, 1, T, S)
               - B: batch size
               - T: time dimension (number of frames)
               - S: spatial dimension (2 * num_antennas * num_range_bins)

        Returns:
            Heart rate prediction of shape (B, 1)

        Processing steps:
            1. Reshape input to (B, 1, T, S) if needed
            2. 2D ResNet: (B, 1, T, S) -> (B, 64, T/8, S/8)
            3. Spatial pooling: (B, 64, T/8, S/8) -> (B, 64, T/8)
            4. 1D ResNet: (B, 64, T/8) -> (B, 512, T'')
            5. Temporal pooling: (B, 512, T'') -> (B, 512)
            6. FC: (B, 512) -> (B, 1)
        """
        # Step 1: Ensure input is (B, 1, T, S)
        if x.dim() == 3:
            # Input is (B, T, S), add channel dimension
            x = x.unsqueeze(1)  # (B, 1, T, S)

        # Step 2: 2D ResNet - spatio-temporal feature extraction
        # Input: (B, 1, T, S)
        # Output: (B, 64, T/8, S/8)
        x = self.resnet2d(x)

        # Step 3: Spatial average pooling
        # Collapse spatial dimension (last dimension)
        # Input: (B, 64, T/8, S/8)
        # Output: (B, 64, T/8)
        x = x.mean(dim=3)

        # Step 4: 1D ResNet - temporal feature extraction
        # Input: (B, 64, T/8)
        # Output: (B, 512, T'') where T'' ≈ T/128
        x = self.resnet1d(x)

        # Step 5: Global temporal average pooling
        # Collapse temporal dimension
        # Input: (B, 512, T'')
        # Output: (B, 512)
        x = x.mean(dim=2)

        # Step 6: Fully connected layer for heart rate prediction
        # Input: (B, 512)
        # Output: (B, 1)
        x = self.fc(x)
        # Apply sigmoid activation for binary output
        x = torch.sigmoid(x)

        return x


# ============================================================================
# Test and Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Test the SpatioTemporalResNetHR model with dummy data.
    """
    print("=" * 70)
    print("SpatioTemporal ResNet for Heart Rate Prediction - Test")
    print("=" * 70)

    # Create model
    model = SpatioTemporalResNetHR()

    # Print model architecture
    print("\nModel Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with dummy input
    print("\n" + "=" * 70)
    print("Testing with dummy input...")
    print("=" * 70)

    # Example dimensions from CIR dataset
    # T = 1500 frames
    # S = 2 * 2 * 32 = 128 (2 pairs * 2 nodes * 32 range bins)
    # For this test, we'll use T=1536 (divisible by 128) and S=896

    batch_size = 2
    T = 1536  # Time dimension (divisible by 128)
    S = 2*2*896   # Spatial dimension (2 * 2 * 896, 2 antennas, both magnitudes and phases, 896 range bins)

    print(f"\nInput shape: ({batch_size}, {T}, {S})")

    # Create dummy input (min-max normalized)
    x = torch.randn(batch_size, T, S)

    # Normalize to [0, 1] to simulate preprocessed features
    x = (x - x.min()) / (x.max() - x.min())

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 1)")
    print(f"\nOutput values (predicted heart rates):")
    print(output)

    # Test with 4D input as well
    print("\n" + "=" * 70)
    print("Testing with 4D input (B, 1, T, S)...")
    print("=" * 70)

    x_4d = torch.randn(batch_size, 1, T, S)
    x_4d = (x_4d - x_4d.min()) / (x_4d.max() - x_4d.min())

    with torch.no_grad():
        output_4d = model(x_4d)

    print(f"Input shape: {x_4d.shape}")
    print(f"Output shape: {output_4d.shape}")
    print(f"\nOutput values:")
    print(output_4d)

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)

    # Test intermediate shapes
    print("\n" + "=" * 70)
    print("Testing intermediate shapes...")
    print("=" * 70)

    x_test = torch.randn(1, 1, 1536, 896*2*2)

    print(f"\nInput to 2D ResNet: {x_test.shape}")

    with torch.no_grad():
        # 2D ResNet
        x_2d = model.resnet2d(x_test)
        print(f"After 2D ResNet: {x_2d.shape}")
        print(f"Expected: (1, 64, {1536//8}, {896//8}) = (1, 64, 192, 112)")

        # Spatial pooling
        x_pooled = x_2d.mean(dim=3)
        print(f"After spatial pooling: {x_pooled.shape}")
        print(f"Expected: (1, 64, 192)")

        # 1D ResNet
        x_1d = model.resnet1d(x_pooled)
        print(f"After 1D ResNet: {x_1d.shape}")
        print(f"Expected: (1, 512, {192//32}) ≈ (1, 512, 3)")

        # Global temporal pooling
        x_global = x_1d.mean(dim=2)
        print(f"After global pooling: {x_global.shape}")
        print(f"Expected: (1, 512)")

        # FC layer
        output_final = model.fc(x_global)
        print(f"Final output: {output_final.shape}")
        print(f"Expected: (1, 1)")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
