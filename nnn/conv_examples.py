"""
Examples demonstrating convolutional neural network layers.

This file shows how to use:
- Conv2D for 2D convolutions
- MaxPool2D and AvgPool2D for spatial downsampling
- BatchNorm2D for normalization
- Dropout for regularization
- Flatten for connecting to dense layers

All layers use NCHW format: (batch, channels, height, width)
"""

import numpy as np
from nnn import Tensor, Dense, Sequential
from nnn.conv import Conv2D, MaxPool2D, AvgPool2D, BatchNorm2D, Dropout, Flatten
from nnn.activations import relu
from nnn.optim import Adam
from nnn.losses import mse_loss


def example_conv2d_basic():
    """Example: Basic Conv2D layer."""
    print("=" * 60)
    print("Basic Conv2D Layer")
    print("=" * 60)
    
    # Create a Conv2D layer: 3 input channels -> 16 output channels
    # Kernel size 3x3, stride 1, padding 1 (same padding)
    conv = Conv2D(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1
    )
    
    print(f"Layer: {conv}")
    print(f"Weight shape: {conv.weight.shape}")
    print(f"  - (out_channels, in_channels, kernel_h, kernel_w)")
    print(f"Bias shape: {conv.bias.shape}")
    
    # Create input: batch of 2 RGB images, 8x8 pixels
    # Shape: (N, C, H, W) = (2, 3, 8, 8)
    input_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
    x = Tensor(input_data, requires_grad=True)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = conv(x)
    
    print(f"Output shape: {output.shape}")
    print(f"  - Same spatial size due to padding=1")
    print()


def example_conv2d_stride_and_padding():
    """Example: Conv2D with different stride and padding."""
    print("=" * 60)
    print("Conv2D with Stride and Padding")
    print("=" * 60)
    
    # Input: 32x32 image with 1 channel
    x = Tensor(np.random.randn(1, 1, 32, 32).astype(np.float32))
    
    # Different configurations
    configs = [
        {"kernel_size": 3, "stride": 1, "padding": 0},  # Output: 30x30
        {"kernel_size": 3, "stride": 1, "padding": 1},  # Output: 32x32 (same)
        {"kernel_size": 3, "stride": 2, "padding": 1},  # Output: 16x16 (downsample)
        {"kernel_size": 5, "stride": 2, "padding": 2},  # Output: 16x16
        {"kernel_size": 7, "stride": 2, "padding": 3},  # Output: 16x16
    ]
    
    print(f"Input shape: {x.shape}")
    print()
    
    for cfg in configs:
        conv = Conv2D(in_channels=1, out_channels=8, **cfg)
        out = conv(x)
        print(f"kernel={cfg['kernel_size']}, stride={cfg['stride']}, "
              f"padding={cfg['padding']} -> Output: {out.shape}")
    
    print("\nFormula: out_size = (in_size + 2*padding - kernel_size) // stride + 1")
    print()


def example_maxpool2d():
    """Example: MaxPool2D layer."""
    print("=" * 60)
    print("MaxPool2D Layer")
    print("=" * 60)
    
    # Create input: 1 image, 1 channel, 8x8
    input_data = np.arange(64).reshape(1, 1, 8, 8).astype(np.float32)
    x = Tensor(input_data, requires_grad=True)
    
    print("Input (8x8):")
    print(x.data[0, 0])
    
    # MaxPool with 2x2 kernel
    pool = MaxPool2D(kernel_size=2, stride=2)
    output = pool(x)
    
    print(f"\nMaxPool2D(kernel_size=2, stride=2)")
    print(f"Output shape: {output.shape}")
    print("Output (4x4 - max values from each 2x2 region):")
    print(output.data[0, 0])
    print()


def example_avgpool2d():
    """Example: AvgPool2D layer."""
    print("=" * 60)
    print("AvgPool2D Layer")
    print("=" * 60)
    
    # Create input: 1 image, 1 channel, 4x4
    input_data = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]).reshape(1, 1, 4, 4).astype(np.float32)
    x = Tensor(input_data, requires_grad=True)
    
    print("Input (4x4):")
    print(x.data[0, 0])
    
    # AvgPool with 2x2 kernel
    pool = AvgPool2D(kernel_size=2, stride=2)
    output = pool(x)
    
    print(f"\nAvgPool2D(kernel_size=2, stride=2)")
    print(f"Output shape: {output.shape}")
    print("Output (2x2 - average of each 2x2 region):")
    print(output.data[0, 0])
    print("\nExample calculation: top-left = (1+2+5+6)/4 = 3.5")
    print()


def example_batchnorm2d():
    """Example: BatchNorm2D layer."""
    print("=" * 60)
    print("BatchNorm2D Layer")
    print("=" * 60)
    
    # Create BatchNorm for 4 channels
    bn = BatchNorm2D(num_features=4, eps=1e-5, momentum=0.1)
    
    print(f"Layer: {bn}")
    print(f"Gamma (scale): {bn.gamma.data}")
    print(f"Beta (shift): {bn.beta.data}")
    
    # Create input: batch of 8, 4 channels, 4x4 spatial
    np.random.seed(42)
    x = Tensor(np.random.randn(8, 4, 4, 4).astype(np.float32), requires_grad=True)
    
    print(f"\nInput shape: {x.shape}")
    
    # Training mode (uses batch statistics)
    bn.train()
    output_train = bn(x)
    
    print(f"Training mode output shape: {output_train.shape}")
    
    # Check normalization: mean should be ~0, std should be ~1 for each channel
    for c in range(4):
        channel_mean = np.mean(output_train.data[:, c, :, :])
        channel_std = np.std(output_train.data[:, c, :, :])
        print(f"  Channel {c}: mean={channel_mean:.4f}, std={channel_std:.4f}")
    
    print(f"\nRunning mean after training: {bn.running_mean}")
    print(f"Running var after training: {bn.running_var}")
    
    # Eval mode (uses running statistics)
    bn.eval()
    output_eval = bn(x)
    print(f"\nEval mode output shape: {output_eval.shape}")
    print()


def example_dropout():
    """Example: Dropout layer."""
    print("=" * 60)
    print("Dropout Layer")
    print("=" * 60)
    
    # Create Dropout with 50% dropout rate
    dropout = Dropout(p=0.5)
    
    print(f"Layer: {dropout}")
    
    # Create input
    np.random.seed(42)
    x = Tensor(np.ones((2, 4, 4, 4)).astype(np.float32), requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Input values: all ones")
    
    # Training mode - applies dropout
    dropout.train()
    output_train = dropout(x)
    
    nonzero = np.count_nonzero(output_train.data)
    total = output_train.data.size
    print(f"\nTraining mode:")
    print(f"  Non-zero elements: {nonzero}/{total} ({100*nonzero/total:.1f}%)")
    print(f"  Non-zero values are scaled by 1/(1-p) = 2.0 for inverted dropout")
    print(f"  Sample values: {output_train.data[0, 0, 0, :4]}")
    
    # Eval mode - no dropout
    dropout.eval()
    output_eval = dropout(x)
    
    print(f"\nEval mode:")
    print(f"  Output equals input (no dropout applied)")
    print(f"  Sample values: {output_eval.data[0, 0, 0, :4]}")
    print()


def example_flatten():
    """Example: Flatten layer."""
    print("=" * 60)
    print("Flatten Layer")
    print("=" * 60)
    
    # Create input: batch of 2, 16 channels, 4x4 spatial
    x = Tensor(np.random.randn(2, 16, 4, 4).astype(np.float32), requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    print(f"  (batch, channels, height, width)")
    
    # Flatten
    flatten = Flatten(start_dim=1)
    output = flatten(x)
    
    print(f"\nFlatten(start_dim=1)")
    print(f"Output shape: {output.shape}")
    print(f"  (batch, channels * height * width)")
    print(f"  16 * 4 * 4 = {16 * 4 * 4}")
    print()


def example_simple_cnn():
    """Example: Simple CNN architecture."""
    print("=" * 60)
    print("Simple CNN Architecture")
    print("=" * 60)
    
    # Build a simple CNN for image classification
    # Input: 1 channel (grayscale), 28x28 image
    
    print("Architecture:")
    print("  Input: (N, 1, 28, 28)")
    print("  Conv2D(1, 32, 3, padding=1) + ReLU -> (N, 32, 28, 28)")
    print("  MaxPool2D(2) -> (N, 32, 14, 14)")
    print("  Conv2D(32, 64, 3, padding=1) + ReLU -> (N, 64, 14, 14)")
    print("  MaxPool2D(2) -> (N, 64, 7, 7)")
    print("  Flatten -> (N, 3136)")
    print("  Dense(3136, 10) -> (N, 10)")
    print()
    
    # Create layers
    conv1 = Conv2D(1, 32, kernel_size=3, padding=1)
    pool1 = MaxPool2D(kernel_size=2, stride=2)
    conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
    pool2 = MaxPool2D(kernel_size=2, stride=2)
    flatten = Flatten()
    fc = Dense(64 * 7 * 7, 10)
    
    # Forward pass
    x = Tensor(np.random.randn(4, 1, 28, 28).astype(np.float32), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Layer by layer
    out = conv1(x)
    out = relu(out)
    print(f"After Conv1 + ReLU: {out.shape}")
    
    out = pool1(out)
    print(f"After Pool1: {out.shape}")
    
    out = conv2(out)
    out = relu(out)
    print(f"After Conv2 + ReLU: {out.shape}")
    
    out = pool2(out)
    print(f"After Pool2: {out.shape}")
    
    out = flatten(out)
    print(f"After Flatten: {out.shape}")
    
    out = fc(out)
    print(f"After FC (logits): {out.shape}")
    print()


def example_cnn_with_batchnorm():
    """Example: CNN with Batch Normalization."""
    print("=" * 60)
    print("CNN with Batch Normalization")
    print("=" * 60)
    
    # Conv -> BatchNorm -> ReLU is a common pattern
    conv = Conv2D(3, 16, kernel_size=3, padding=1)
    bn = BatchNorm2D(16)
    
    print("Pattern: Conv2D -> BatchNorm2D -> ReLU")
    print()
    
    # Create input
    x = Tensor(np.random.randn(8, 3, 16, 16).astype(np.float32), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    out = conv(x)
    print(f"After Conv2D: {out.shape}")
    
    out = bn(out)
    print(f"After BatchNorm2D: {out.shape}")
    
    out = relu(out)
    print(f"After ReLU: {out.shape}")
    
    # Check that BN normalized the data
    print(f"\nBatchNorm stats (after BN, before ReLU):")
    # Note: We use the pre-ReLU values for meaningful stats
    bn_out = bn(conv(x))
    for c in range(min(4, 16)):
        channel_mean = np.mean(bn_out.data[:, c, :, :])
        channel_std = np.std(bn_out.data[:, c, :, :])
        print(f"  Channel {c}: mean={channel_mean:.4f}, std={channel_std:.4f}")
    print()


def example_conv2d_backward():
    """Example: Conv2D backward pass (gradient computation)."""
    print("=" * 60)
    print("Conv2D Backward Pass (Gradient Computation)")
    print("=" * 60)
    
    # Create a simple conv layer
    conv = Conv2D(1, 2, kernel_size=3, padding=1)
    
    # Create input
    np.random.seed(42)
    x = Tensor(np.random.randn(1, 1, 4, 4).astype(np.float32), requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {conv.weight.shape}")
    
    # Forward pass
    output = conv(x)
    print(f"Output shape: {output.shape}")
    
    # Create a simple loss (sum of outputs)
    loss = output.sum()
    print(f"\nLoss (sum of outputs): {loss.data}")
    
    # Backward pass
    loss.backward()
    
    print(f"\nGradients computed:")
    print(f"  Weight gradient shape: {conv.weight.grad.data.shape}")
    print(f"  Bias gradient shape: {conv.bias.grad.data.shape}")
    print(f"  Bias gradient values: {conv.bias.grad.data}")
    print()


def example_cnn_training_step():
    """Example: Single CNN training step."""
    print("=" * 60)
    print("CNN Training Step")
    print("=" * 60)
    
    # Create a simple CNN
    conv = Conv2D(1, 8, kernel_size=3, padding=1)
    pool = MaxPool2D(kernel_size=2)
    flatten = Flatten()
    fc = Dense(8 * 4 * 4, 2)  # 8 channels, 4x4 after pooling from 8x8
    
    # Collect all parameters
    params = conv.parameters() + fc.parameters()
    optimizer = Adam(params, lr=0.01)
    
    # Create dummy data
    np.random.seed(42)
    x = Tensor(np.random.randn(4, 1, 8, 8).astype(np.float32), requires_grad=True)
    target = Tensor(np.random.randn(4, 2).astype(np.float32))
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {target.shape}")
    
    # Training loop (3 steps)
    for step in range(3):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        out = conv(x)
        out = relu(out)
        out = pool(out)
        out = flatten(out)
        pred = fc(out)
        
        # Compute loss
        loss = mse_loss(pred, target)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        print(f"Step {step + 1}: Loss = {loss.data.item():.4f}")
    
    print()


if __name__ == "__main__":
    # Run all examples
    example_conv2d_basic()
    example_conv2d_stride_and_padding()
    example_maxpool2d()
    example_avgpool2d()
    example_batchnorm2d()
    example_dropout()
    example_flatten()
    example_simple_cnn()
    example_cnn_with_batchnorm()
    example_conv2d_backward()
    example_cnn_training_step()
    
    print("=" * 60)
    print("All CNN examples completed!")
    print("=" * 60)
