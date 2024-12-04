import torch
from torch import nn
from einops import einsum, repeat, rearrange
from math import sqrt


class MaskedSum(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x, mask):
        # x: [B, N, D]
        # mask: [B, K, N]
        return einsum(x, mask, 'b n d, b k n -> b k d')
    

def get_abs_grid(num_patches):
    width = int(sqrt(num_patches))
    offsets = torch.linspace(-1, 1, width)
    xy_coords = torch.stack(torch.meshgrid(offsets, offsets, indexing='ij'), dim=-1)
    xy_coords = rearrange(xy_coords, 'h w d -> (h w) d')
    return  xy_coords