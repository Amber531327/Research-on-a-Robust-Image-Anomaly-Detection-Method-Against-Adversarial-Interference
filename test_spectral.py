"""
频域过滤模块测试脚本
"""
import torch
import sys
sys.path.insert(0, r'e:\Graduation_Project\Improvedlast_PatchGuard')
from spectral_filter import SpectralGatingModule, MultiLayerSpectralFilter

# Test 1: SpectralGatingModule
print('Test 1: SpectralGatingModule')
module = SpectralGatingModule(embed_dim=384, num_patches=256)
x = torch.randn(2, 256, 384)
out = module(x)
print(f'  Input shape: {x.shape}')
print(f'  Output shape: {out.shape}')
assert x.shape == out.shape, 'Shape mismatch!'

# Get regularization loss
reg_loss = module.get_regularization_loss()
print(f'  Regularization loss: {reg_loss:.6f}')
print('  SpectralGatingModule test PASSED!')

# Test 2: MultiLayerSpectralFilter
print('\nTest 2: MultiLayerSpectralFilter')
multi_filter = MultiLayerSpectralFilter(
    layer_indices=[3, 6],
    embed_dim=384,
    num_patches=256,
    gate_reduction=4,
    regularization='l1',
    reg_weight=0.01,
    device='cpu'
)

# Test filtering
x = torch.randn(2, 256, 384)
out3 = multi_filter.filter(x, layer_idx=3)
out6 = multi_filter.filter(x, layer_idx=6)
out9 = multi_filter.filter(x, layer_idx=9)  # Should return unchanged

print(f'  Layer 3 filter applied: {3 in multi_filter}')
print(f'  Layer 6 filter applied: {6 in multi_filter}')
print(f'  Layer 9 filter applied: {9 in multi_filter}')

total_reg = multi_filter.get_total_regularization_loss()
print(f'  Total regularization loss: {total_reg:.6f}')

print('  MultiLayerSpectralFilter test PASSED!')

print('\n=== All tests passed! ===')
