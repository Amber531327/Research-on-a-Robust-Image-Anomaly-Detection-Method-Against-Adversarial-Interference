"""Test script for plug-and-play modules: MultiScaleFeatureFusion + SpectralFilter"""
import torch
import sys
sys.path.insert(0, '.')
from patchguard import PatchGuard, MultiScaleFeatureFusion
import argparse

def make_args(**overrides):
    """Create a default args namespace with optional overrides"""
    defaults = dict(
        hf_path='vit_small_patch14_dinov2.lvd142m',
        feature_layers=[3, 6, 9, 12],
        reg_layers=[3, 6, 9, 12],
        image_size=224,
        hidden_dim=2048,
        dsc_layers=1,
        dsc_heads=4,
        fusion_type='gated_attention',
        fusion_dim=384,
        spectral_filter_layers=[3, 6],
        spectral_gate_reduction=4,
        spectral_reg_type='l1',
        spectral_reg_weight=0.01,
        use_multiscale_fusion=True,
        use_spectral_filter=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_config(name, use_fusion, use_spectral):
    """Test a specific plug-and-play configuration"""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  use_multiscale_fusion={use_fusion}, use_spectral_filter={use_spectral}")
    print(f"{'='*60}")
    
    args = make_args(use_multiscale_fusion=use_fusion, use_spectral_filter=use_spectral)
    model = PatchGuard(args, 'cpu')
    
    print(f"  - Feature dim: {model.feature_extractor.embed_dim}")
    print(f"  - Num patches: {model.num_patches}")
    print(f"  - Fusion module: {'YES' if model.feature_extractor.fusion is not None else 'NO'}")
    print(f"  - Spectral filter: {'YES' if model.feature_extractor.spectral_filter is not None else 'NO'}")
    print(f"  - Trainable params: {count_trainable_params(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"  - Output shape: {output.shape}")
    
    # Test with regularization (training mode)
    features, attn_weights, spectral_loss, _, _ = model.feature_extractor(x, use_reg=True)
    print(f"  - Features shape: {features.shape}")
    print(f"  - Attention weight tensors: {len(attn_weights)}")
    print(f"  - Spectral loss: {spectral_loss}")
    
    if not use_spectral:
        assert spectral_loss == 0.0, f"Spectral loss should be 0.0 when disabled, got {spectral_loss}"
    
    if not use_fusion:
        assert model.feature_extractor.fusion is None, "Fusion module should be None when disabled"
        assert model.feature_extractor.embed_dim == model.feature_extractor.base_embed_dim
    
    print(f"  ✓ Test passed!")
    return count_trainable_params(model)

# Run all four combinations
print("=" * 60)
print("Testing Plug-and-Play Module Configurations")
print("=" * 60)

params = {}
params['both_on']     = test_config("Both ON (default)",          use_fusion=True,  use_spectral=True)
params['fusion_only'] = test_config("Fusion ON, Spectral OFF",    use_fusion=True,  use_spectral=False)
params['spectral_only'] = test_config("Fusion OFF, Spectral ON",  use_fusion=False, use_spectral=True)
params['both_off']    = test_config("Both OFF (baseline)",         use_fusion=False, use_spectral=False)

# Summary
print(f"\n{'='*60}")
print("Parameter Count Comparison")
print(f"{'='*60}")
for name, count in params.items():
    print(f"  {name:20s}: {count:>10,} params")

print(f"\n{'='*60}")
print("=== All 4 plug-and-play tests passed! ===")
print(f"{'='*60}")
