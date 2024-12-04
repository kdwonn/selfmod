from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from einops import rearrange, repeat, einsum
from typing import Union, Callable, Optional, Tuple

from ocl.attention import SlotAttention
from ocl.neural_networks import build_two_layer_mlp, Sequential, DummyPositionEmbed
from ocl.slot_dict import BlockGRU, BlockLinear, BlockLayerNorm
from ocl.slot_dict import VectorQuantize, ResidualVQ, FSQ, DummyQ, MemDPC
import ocl.typing


class SlotAttentionIterCA(SlotAttention):
    """Implementation of SlotAttention with feedback."""
    def __init__(
        self,
        dim: int,
        feature_dim: int,
        pool: Union[VectorQuantize, ResidualVQ, FSQ, MemDPC],
        ca: torch.nn.MultiheadAttention,
        pos_embed: nn.Module,
        input_feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        feedback_normalization: bool = False,
        num_blocks: int = 1,
        scale_sum_to_one: bool = False,
        feedback_scale: float = 1,
        pool_ln: bool = False,
        pool_out_ln: bool = False,
        ca_takes_attn: bool = False,
        feedback_before_pos: bool = True,
    ):
        super().__init__(
            dim,
            feature_dim,
            kvq_dim,
            n_heads,
            iters,
            eps,
            ff_mlp,
            use_projection_bias,
            use_implicit_differentiation,
        )
        self.feedback_normalization = feedback_normalization
        self.pool = pool
        self.ca = ca
        self.feedback_scale = feedback_scale
        self.scale_sum_to_one = scale_sum_to_one

        self.feedback_before_pos = feedback_before_pos
        self.ca_pool_norm = nn.LayerNorm(dim) if pool_ln else nn.Identity()
        self.ca_slot_norm = nn.LayerNorm(dim) if pool_out_ln else nn.Identity()
        self.ca_feat_norm = nn.LayerNorm(input_feature_dim if self.feedback_before_pos else feature_dim)
        self.pos_embed = pos_embed
        
        if self.scale_sum_to_one:
            assert self.feedback_scale < 1            
        if num_blocks > 1:
            self.gru = BlockGRU(self.kvq_dim, dim, k=num_blocks)
            self.norm_slots = BlockLayerNorm(dim, k=num_blocks)
            self.ff_mlp = build_two_layer_mlp(dim, dim, 4*dim, initial_layer_norm=True, residual=True, num_blocks=num_blocks)

        self.ca_takes_attn = ca_takes_attn

    def feedback(self, inputs, slots, attn):
        slots = self.ca_pool_norm(slots)
        qslots, inds, commit = self.pool(slots)
        qslots = self.ca_slot_norm(qslots)
        
        if self.ca_takes_attn:
            td_sig = self.ca(
                inputs=inputs,
                codes=qslots,
                masks=attn
            )
            ca_attn = attn
        else:
            td_sig, ca_attn = self.ca(
                query=self.ca_feat_norm(inputs), 
                key=qslots, 
                value=qslots, 
                need_weights=True
            )
            ca_attn = rearrange(ca_attn, 'b n k -> b k n')

        if self.scale_sum_to_one:
            patched_inputs = (1-self.feedback_scale)*inputs + self.feedback_scale*td_sig
        else:
            patched_inputs = inputs + self.feedback_scale*td_sig

        return patched_inputs, ca_attn, inds, commit

    def iterate(self, slots, inputs, pos, iters, masks=None):
        commits = []
        attns = []
        b, n, _ = inputs.shape

        if not self.feedback_before_pos:
            inputs = self.pos_embed(inputs, pos)

        for i in range(iters):
            if self.feedback_before_pos:
                pos_inputs = self.pos_embed(inputs, pos)
            else:
                pos_inputs = inputs

            k_inputs = v_inputs = self.norm_input(pos_inputs)

            k = self.to_k(k_inputs).view(b, n, self.n_heads, self.dims_per_head)
            v = self.to_v(v_inputs).view(b, n, self.n_heads, self.dims_per_head)
            
            slots, attn = self.step(slots, k, v, masks)
            attns += [attn]

            inputs, ca_attn, inds, commit = self.feedback(inputs, slots, attn)
            commits += [commit]
        
        pool_stats = {
            'indices': inds,
            'commit': torch.mean(torch.stack(commits), dim=0),
        }
        
        return slots, inputs, attns, pool_stats, ca_attn

    def forward(
        self, inputs: ocl.typing.FeatureExtractorOutput, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        slots = conditioning
        inputs, pos = inputs.features, inputs.positions

        if self.use_implicit_differentiation:
            slots, inputs, attn, pool_stats, ca_attn = self.iterate(
                slots, inputs, pos, self.iters, masks)
            slots, inputs, attn, pool_stats, ca_attn = self.iterate(
                slots.detach(), inputs, pos, 1, masks)
        else:
            slots, inputs, attn, pool_stats, ca_attn = self.iterate(
                slots, inputs, pos, self.iters, masks)

        return slots, inputs, attn, pool_stats, ca_attn


class SlotAttentionGroupingIterCA(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        input_feature_dim: int,
        pool: Union[VectorQuantize, ResidualVQ, FSQ],
        ca: nn.Module,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        num_blocks: int = 1,
        scale_sum_to_one: bool = False,
        feedback_scale: float = 1,
        pool_ln: bool = False,
        pool_out_ln: bool = False,
        ca_takes_attn: bool = False,
        feedback_before_pos: bool = True,
    ):
        """Initialize Slot Attention Grouping.

        Args:
            pool: slot pool to use: choose among VQ, RVQ, FSQ
            pool_config: slot pool configuration
            **kwargs: other arguments for SlotAttentionGrouping
        """
        super().__init__()
        self._object_dim = object_dim
        self.slot_attention = SlotAttentionIterCA(
            dim=object_dim,
            feature_dim=feature_dim,
            pool=pool,
            ca=ca,
            pos_embed=positional_embedding,
            input_feature_dim=input_feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
            num_blocks=num_blocks,
            scale_sum_to_one=scale_sum_to_one,
            feedback_scale=feedback_scale,
            pool_ln=pool_ln,
            pool_out_ln=pool_out_ln,
            ca_takes_attn=ca_takes_attn,
            feedback_before_pos=feedback_before_pos,
        )

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

    @property
    def object_dim(self):
        return self._object_dim
    
    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        
        slots, feature, attn, pool_stats, ca_attn = self.slot_attention(feature, conditioning, slot_mask)

        indices, commit = pool_stats['indices'], pool_stats['commit']

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingOutput(
            slots, 
            feature_attributions=attn[-1],
            feature_attributions_2nd = attn[-2], 
            feature_attributions_1st = attn[-3], 
            is_empty=slot_mask,
            pool_indices=indices,
            commit_loss=commit.sum(),
            alpha_weights=ca_attn,
            pos_feat = feature,
        )