import torch
import torch.nn as nn 
import torch.nn.functional as F
import dataclasses
from torchtyping import TensorType
from typing import Callable, Dict, Optional, Tuple, Union, OrderedDict

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
        
@dataclasses.dataclass
class CrossAttnOutput:
    reconstruction: TensorType["batch_size", "n_patches", "n_patch_features"]  # noqa: F821
    masks: TensorType["batch_size", "n_objects", "n_patches"]  # noqa: F821
    masks_as_image: Optional[
        TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    ] = None
    target: Optional[TensorType["batch_size", "n_patches", "n_patch_features"]] = None  # noqa: F821

class CrossAttn(nn.Module):
    def __init__(
        self, 
        target_feature_dim: int,
        slot_feature_dim: int,
        n_heads: int = 1,
        residual: bool = True,
        mlp_ratio: int = 0
    ):
        super(CrossAttn, self).__init__()
        self.slot2target_projection = nn.Linear(slot_feature_dim, target_feature_dim)
        
        self.ln_before_ca_slots = LayerNorm(target_feature_dim)
        self.ln_before_ca_target = LayerNorm(target_feature_dim)
        self.ca = nn.MultiheadAttention(target_feature_dim, n_heads)
        self.cross_attn = self.residual_cross_attn if residual else self.non_residual_cross_attn

        mlp_dim = int(target_feature_dim*mlp_ratio)
        self.ln_before_mlp = LayerNorm(target_feature_dim) if mlp_ratio != 0 else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(target_feature_dim, mlp_dim)),
                ('ln', LayerNorm(mlp_dim)),
                ("gelu", nn.GELU()),
                ("c_proj", nn.Linear(mlp_dim, target_feature_dim))
            ])) if mlp_ratio != 0 else nn.Identity()
        self.multi_layer_perceptron = self.residual_mlp if residual and mlp_ratio != 0 else self.non_residual_mlp

    def non_residual_mlp(
        self,
        td_signal
    ):
        return self.mlp(self.ln_before_mlp(td_signal))
    
    def residual_mlp(
        self,
        td_signal
    ):
        return td_signal + self.mlp(self.ln_before_mlp(td_signal))

    def non_residual_cross_attn(
        self,
        target,
        slots
    ):
        permuted_slots = self.ln_before_ca_slots(slots).permute(1,0,2)
        permuted_target = self.ln_before_ca_target(target).permute(1,0,2)
        td_signal, attention_maps = self.ca(permuted_target, permuted_slots, permuted_slots)
        return td_signal.permute(1,0,2), attention_maps.permute(0,2,1)
    
    def residual_cross_attn(
        self,
        target,
        slots
    ):
        permuted_slots = self.ln_before_ca_slots(slots).permute(1,0,2)
        permuted_target = self.ln_before_ca_target(target).permute(1,0,2)
        td_signal, attention_maps = self.ca(permuted_target, permuted_slots, permuted_slots)
        return target + td_signal.permute(1,0,2), attention_maps.permute(0,2,1)

    def forward(
        self,
        target,
        slots
    ) -> CrossAttnOutput:
        slots = self.slot2target_projection(slots)
        td_signal, attention_maps = self.cross_attn(target, slots)
        td_signal = self.multi_layer_perceptron(td_signal)
        return CrossAttnOutput(reconstruction=td_signal, 
                                masks=attention_maps)


