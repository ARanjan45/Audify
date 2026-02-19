import torch
import pytest
from model import AudioCNN

def test_model_output_shape():
    """Model should output (batch_size, num_classes) tensor."""
    model = AudioCNN(num_classes=50)
    model.eval()
    dummy_input = torch.randn(4, 1, 128, 431)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (4, 50), f"Expected (4, 50), got {output.shape}"

def test_model_output_is_finite():
    """Model output should not contain NaN or Inf values."""
    model = AudioCNN(num_classes=50)
    model.eval()
    dummy_input = torch.randn(4, 1, 128, 431)
    with torch.no_grad():
        output = model(dummy_input)
    assert torch.isfinite(output).all(), "Model output contains NaN or Inf"

def test_feature_map_extraction():
    """Model should return feature maps when return_feature_maps=True."""
    model = AudioCNN(num_classes=50)
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 431)
    with torch.no_grad():
        output, feature_maps = model(dummy_input, return_feature_maps=True)
    assert output.shape == (1, 50)
    assert "conv1" in feature_maps
    assert "layer1" in feature_maps
    assert "layer4" in feature_maps

def test_residual_block_shapes():
    """Residual blocks should preserve spatial dims when stride=1."""
    from model import ResidualBlock
    block = ResidualBlock(64, 64, stride=1)
    x = torch.randn(2, 64, 32, 32)
    out = block(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
