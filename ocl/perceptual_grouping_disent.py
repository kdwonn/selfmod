import math
from typing import Any, Dict, Optional
from functools import partial

import numpy
import torch
from sklearn import cluster
from torch import nn
from typing import Union, Callable, Optional, Tuple
from ema_pytorch import EMA
from torch.nn.modules import Module
from einops import rearrange, repeat
from contextlib import nullcontext

import ocl.typing
from ocl.attention import *


def detach(x: torch.Tensor, condition: bool = True) -> torch.Tensor:
    """Detach tensor."""
    return x.detach() if condition else x


class SlotAttentionGroupingDisent(nn.Module):
    """Implementation of SlotAttention for perceptual grouping."""

    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        position_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        num_blocks=1,
    ):
        """Initialize Slot Attention Grouping.

        Args:
            position_dim: dimension of positional feature
        """
        super().__init__()
        self._object_dim = object_dim
        self.slot_attention = SlotAttentionMultiBlock(
            dim=object_dim + position_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
            num_blocks=num_blocks,
        )

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

        self.position_dim = position_dim

    @property
    def object_dim(self):
        return self._object_dim

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        """Apply slot attention based perceptual grouping.

        Args:
            feature: Features used for grouping.
            conditioning: Initial conditioning vectors for slots.
            slot_mask: Slot mask where true indicates that the slot should be masked.

        Returns:
            The grouped features.
        """
        if self.positional_embedding:
            feature = self.positional_embedding(feature.features, feature.positions)
        else:
            feature = feature.features

        slots, attn = self.slot_attention(feature, conditioning, slot_mask)

        object_slots, position_slots = slots[..., :self._object_dim], slots[..., -self.position_dim:]

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingDisentOutput(
            object_slots, position_slots, feature_attributions=attn, is_empty=slot_mask
        )
    

class SlotAttentionGroupingInvTransScale(nn.Module):
    """Implementation of SlotAttention for perceptual grouping."""

    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int] = None,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        scale_factor: int = 5,
        use_weighted_avg: bool = True,
        sg_pos_scale: bool = False,
        off_amp: bool = True
    ):
        """Initialize Slot Attention Grouping.

        Args:
            position_dim: dimension of positional feature
        """
        super().__init__()
        self._object_dim = object_dim
        self.position_dim = 4
        self.slot_attention = SlotAttentionInvTransScale(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
            scale_factor=scale_factor,
            use_weighted_avg=use_weighted_avg,
            sg_pos_scale=sg_pos_scale,
            off_amp=off_amp
        )

        self.positional_embedding = positional_embedding

    @property
    def object_dim(self):
        return self._object_dim

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        """Apply slot attention based perceptual grouping.

        Args:
            feature: Features used for grouping.
            conditioning: Initial conditioning vectors for slots.
            slot_mask: Slot mask where true indicates that the slot should be masked.

        Returns:
            The grouped features.
        """
        if self.positional_embedding:
            feature = self.positional_embedding(feature.features, feature.positions)
        else:
            feature = feature.features

        slots, attn, rel_grid = self.slot_attention(feature, conditioning, slot_mask)

        object_slots, position_slots = slots[..., :self._object_dim], slots[..., -self.position_dim:]

        return ocl.typing.PerceptualGroupingDisentOutput(
            object_slots, position_slots, feature_attributions=attn, is_empty=slot_mask, rel_pos_grid=rel_grid.abs().sum(dim=-1)
        )