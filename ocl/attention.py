from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from einops import rearrange, repeat, einsum
from typing import Union, Callable, Optional, Tuple

from ocl.neural_networks import build_two_layer_mlp, Sequential, DummyPositionEmbed
from ocl.slot_dict import BlockGRU, BlockLinear, BlockLayerNorm
from ocl.slot_dict import VectorQuantize, ResidualVQ, FSQ, DummyQ, MemDPC
from ocl.synthesizer import BraodcastingSynthesizer, MaskedBroadcastingSynthesizer, SynthPrompt, DivergenceSynthesizer, SoftMaskedBroadcastingSynthesizer
import ocl.typing
from ocl.hypernet import HypernetOutput, Hypernet
from ocl.utils.misc import get_abs_grid


class SlotAttention(nn.Module):
    """Implementation of SlotAttention.

    Based on the slot attention implementation of Phil Wang available at:
    https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.use_implicit_differentiation = use_implicit_differentiation

        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim

        if self.kvq_dim % self.n_heads != 0:
            raise ValueError("Key, value, query dimensions must be divisible by number of heads.")
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head**-0.5

        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        self.gru = nn.GRUCell(self.kvq_dim, dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.ff_mlp = ff_mlp

    def step(self, slots, k, v, masks=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            # Masked slots should not take part in the competition for features. By replacing their
            # dot-products with -inf, their attention values will become zero within the softmax.
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)

        if self.ff_mlp:
            slots = self.ff_mlp(slots)

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, k, v, masks=None):
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, masks)
        return slots, attn

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks)
            slots, attn = self.step(slots.detach(), k, v, masks)
        else:
            slots, attn = self.iterate(slots, k, v, masks)

        return slots, attn


class SlotAttentionAttnMod(SlotAttention):
    def __init__(self, *args, mod_weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.mod_weight = mod_weight
        assert 0 <= self.mod_weight <= 1

    def step(self, slots, k, v, masks=None, attn_mod=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        
        if attn_mod is not None:
            attn_mod = attn_mod.unsqueeze(2)
            attn = attn * (1 - self.mod_weight) + attn_mod * self.mod_weight

        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)

        if self.ff_mlp:
            slots = self.ff_mlp(slots)

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, k, v, masks=None, attn_mod=None):
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, masks, attn_mod)
        return slots, attn

    def forward(
        self, 
        inputs: torch.Tensor, 
        conditioning: torch.Tensor, 
        masks: Optional[torch.Tensor] = None,
        attn_mod: Optional[torch.Tensor] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks, attn_mod)
            slots, attn = self.step(slots.detach(), k, v, masks, attn_mod)
        else:
            slots, attn = self.iterate(slots, k, v, masks, attn_mod)

        return slots, attn


class SlotAttentionMultiBlock(SlotAttention):
    def __init__(
        self, 
        dim: int, 
        feature_dim: int, 
        kvq_dim: Optional[int] = None, 
        n_heads: int = 1, 
        iters: int = 3, 
        eps: float = 1e-8, 
        ff_mlp: Optional[nn.Module] = None, 
        use_projection_bias: bool = False, 
        use_implicit_differentiation: bool = False, 
        num_blocks: int = 1
    ):
        super().__init__(dim, feature_dim, kvq_dim, n_heads, iters, eps, ff_mlp, use_projection_bias, use_implicit_differentiation)

        self.num_blocks = num_blocks
        if self.num_blocks > 1:
            self.gru = BlockGRU(self.kvq_dim, dim, k=num_blocks)
            self.norm_slots = BlockLayerNorm(dim, k=num_blocks)
            self.ff_mlp = build_two_layer_mlp(dim, dim, 4*dim, initial_layer_norm=True, residual=True, num_blocks=num_blocks)


class SlotAttentionWithFeedback(SlotAttention):
    """Implementation of SlotAttention with feedback."""
    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        feedback_normalization: bool = False,
        num_blocks: int = 1,
        feedback_scale: float = 1.0,
        feedback_gating: bool = False,
        scale_sum_to_one: bool = False,
        feedback_slotwise: bool = False,
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
        self.num_blocks = num_blocks
        if self.num_blocks > 1:
            self.gru = BlockGRU(self.kvq_dim, dim, k=num_blocks)
            self.norm_slots = BlockLayerNorm(dim, k=num_blocks)
            self.ff_mlp = build_two_layer_mlp(dim, dim, 4*dim, initial_layer_norm=True, residual=True, num_blocks=num_blocks)
        
        self.feedback_scale = feedback_scale
        self.gate = nn.Identity()
        if feedback_gating:
            self.feedback_scale = nn.Parameter(torch.zeros(1))
            self.gate = nn.Tanh()
        self.scale_sum_to_one = scale_sum_to_one

        self.feedback_slotwise = feedback_slotwise
    
    def add_feedback(self, x, feedback, condition):
        if not condition:
            return x
        
        if self.feedback_slotwise:
            x = repeat(x, 'b n d -> b k n d', k=feedback.shape[1])
        
        scale = self.gate(self.feedback_scale)
        if self.scale_sum_to_one:
            assert scale <= 1
            return (1-scale)*x + scale*feedback
        else:
            return x + feedback
        
    def step(self, slots, k, v, masks=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        #  added for the slotwise fereedback, rest code is same as the original step
        if len(v.shape) == 5:
            updates = torch.einsum("bijhd,bihj->bihd", v, attn)
        else:
            updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)

        if self.ff_mlp:
            slots = self.ff_mlp(slots)

        return slots, attn_before_reweighting.mean(dim=2)
            
    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None, td_signal: Optional[ocl.typing.SynthesizerOutput] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        feedback = None
        if td_signal is not None:
            feedback_type = td_signal.feedback_type
            feedback = td_signal.feedback

        k_inputs = v_inputs = self.norm_input(inputs)
        if td_signal is not None:
            if self.feedback_normalization:
                feedback = self.norm_input(feedback)
            k_inputs = self.add_feedback(k_inputs, feedback, 'k' in feedback_type)
            v_inputs = self.add_feedback(v_inputs, feedback, 'v' in feedback_type)

        head_pattern = '... (h d) -> ... h d' 
        k = rearrange(self.to_k(k_inputs), head_pattern, h=self.n_heads)
        v = rearrange(self.to_v(v_inputs), head_pattern, h=self.n_heads)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks)
            slots, attn = self.step(slots.detach(), k, v, masks)
        else:
            slots, attn = self.iterate(slots, k, v, masks)

        stats = {
            'k_inputs': k_inputs,
            'v_inputs': v_inputs,
            'pos_feedback': feedback,
        }

        return slots, attn, stats


class SlotAttentionWithFeedbackHyper(SlotAttention):
    """Implementation of SlotAttention with feedback."""
    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        num_blocks: int = 1,
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
        self.num_blocks = num_blocks
        if self.num_blocks > 1:
            self.gru = BlockGRU(self.kvq_dim, dim, k=num_blocks)
            self.norm_slots = BlockLayerNorm(dim, k=num_blocks)
            self.ff_mlp = build_two_layer_mlp(dim, dim, 4*dim, initial_layer_norm=True, residual=True, num_blocks=num_blocks)

    def step(self, slots, k, v, cond_adapter, masks=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        #  added for the slotwise fereedback, rest code is same as the original step
        if len(v.shape) == 5:
            updates = torch.einsum("bijhd,bihj->bihd", v, attn)
        else:
            updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)

        if self.ff_mlp:
            slots = self.ff_mlp(slots)

        if cond_adapter is not None:
            slots = cond_adapter(slots)

        return slots, attn_before_reweighting.mean(dim=2)

    def conditional_layer_norm(self, x, ln_w, ln_b):
        normalized = F.layer_norm(x, x.shape[-1:])
        ln_b = rearrange(ln_b, 'b k d -> b k 1 d')
        return einsum(normalized, ln_w, 'b n d, b k d -> b k n d') + ln_b

    def conditional_adapter(self, x, up_proj, down_proj, ln_w, ln_b):
        normalized = F.layer_norm(x, x.shape[-1:])
        normalized = normalized * ln_w + ln_b
        down_projected = einsum(normalized, down_proj, 'b k d, b d d_down -> b k d_down')
        up_projected = einsum(F.gelu(down_projected), up_proj, 'b k d_down, b d_down d -> b k d')
        return up_projected + x
    
    def iterate(self, slots, k, v, cond_adapter, masks=None):
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, cond_adapter, masks)
        return slots, attn

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None, td_signal: Optional[HypernetOutput] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        k_inputs = self.norm_input(inputs)
        if td_signal is None:
            v_inputs = self.norm_input(inputs)
            cond_adapter = None
        else:
            v_inputs = self.conditional_layer_norm(inputs, td_signal.ln_w, td_signal.ln_b)
            if td_signal.use_ada:
                cond_adapter = partial(
                    self.conditional_adapter,
                    up_proj=td_signal.ada_up_proj,
                    down_proj=td_signal.ada_down_proj,
                    ln_w=td_signal.ada_ln_w,
                    ln_b=td_signal.ada_ln_b,
                )
            else:
                cond_adapter = None

        head_pattern = '... (h d) -> ... h d' 
        k = rearrange(self.to_k(k_inputs), head_pattern, h=self.n_heads)
        v = rearrange(self.to_v(k_inputs), head_pattern, h=self.n_heads)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, cond_adapter, masks)
            slots, attn = self.step(slots.detach(), k, v, cond_adapter, masks)
        else:
            slots, attn = self.iterate(slots, k, v, cond_adapter, masks)

        stats = {
            'k_inputs': k_inputs,
            'v_inputs': v_inputs,
            'ln_w': td_signal.ln_w if td_signal is not None else None,
            'ln_b': td_signal.ln_b if td_signal is not None else None,
        }

        return slots, attn, stats


class SlotAttentionWithFeedbackIterQuant(SlotAttention):
    """Implementation of SlotAttention with feedback."""
    def __init__(
        self,
        dim: int,
        feature_dim: int,
        pool: Union[VectorQuantize, ResidualVQ, FSQ, MemDPC],
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        feedback_normalization: bool = False,
        num_blocks: int = 1,
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
        self.num_blocks = num_blocks
        if self.num_blocks > 1:
            self.gru = BlockGRU(self.kvq_dim, dim, k=num_blocks)
            self.norm_slots = BlockLayerNorm(dim, k=num_blocks)
            self.ff_mlp = build_two_layer_mlp(dim, dim, 4*dim, initial_layer_norm=True, residual=True, num_blocks=num_blocks)

    def iterate(self, slots, k, v, is_feedbacked, masks=None):
        commits = []
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, masks)
            if is_feedbacked:
                inds = None
                commit = torch.zeros(1).cuda()
            else:
                slots, inds, commit = self.pool(slots)
        commits += [commit]
        return slots, attn, inds, torch.mean(torch.stack(commits), dim=0)

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None, td_signal: Optional[ocl.typing.SynthesizerOutput] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning
        is_feedbacked = td_signal is not None

        v_feedback, k_feedback = False, False
        feedback = None
        if is_feedbacked:
            v_feedback = 'v' in td_signal.feedback_type
            k_feedback = 'k' in td_signal.feedback_type
            feedback = td_signal.feedback

        k_inputs = v_inputs = self.norm_input(inputs)
        if is_feedbacked:
            if self.feedback_normalization:
                k_inputs = self.norm_input(inputs if not k_feedback else inputs+feedback)
                v_inputs = self.norm_input(inputs if not v_feedback else inputs+feedback)
            else:
                k_inputs = k_inputs if not k_feedback else k_inputs+feedback
                v_inputs = v_inputs if not v_feedback else v_inputs+feedback

        k = self.to_k(k_inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(v_inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, attn, inds, commit = self.iterate(slots, k, v, is_feedbacked, masks)
            slots, attn = self.step(slots.detach(), k, v, masks)
            if not is_feedbacked:
                slots, inds, last_commit = self.pool(slots)
                commit = (commit + last_commit) / 2
        else:
            slots, attn, inds, commit = self.iterate(slots, k, v, is_feedbacked, masks)

        stats = {
            'k_inputs': k_inputs,
            'v_inputs': v_inputs,
            'pos_feedback': feedback,
        }
        pool_stats = {
            'indices': inds,
            'commit': commit.sum(),
        }

        return slots, attn, stats, pool_stats


class SlotAttentionWithFeedbackIterQuantSynth(SlotAttention):
    """Implementation of SlotAttention with feedback."""
    def __init__(
        self,
        dim: int,
        feature_dim: int,
        pool: Union[VectorQuantize, ResidualVQ, FSQ, MemDPC],
        synth: Union[BraodcastingSynthesizer, MaskedBroadcastingSynthesizer],
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        feedback_normalization: bool = False,
        num_blocks: int = 1,
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
        self.synth = synth
        if num_blocks > 1:
            self.gru = BlockGRU(self.kvq_dim, dim, k=num_blocks)
            self.norm_slots = BlockLayerNorm(dim, k=num_blocks)
            self.ff_mlp = build_two_layer_mlp(dim, dim, 4*dim, initial_layer_norm=True, residual=True, num_blocks=num_blocks)

    def iterate(self, slots, k_inputs, v_inputs, masks=None):
        commits = []
        b, n, _ = k_inputs.shape
        v_feedback = torch.zeros_like(v_inputs).cuda()
        for i in range(self.iters):
            k = self.to_k(k_inputs).view(b, n, self.n_heads, self.dims_per_head)
            v = self.to_v(v_inputs + v_feedback).view(b, n, self.n_heads, self.dims_per_head)
            slots, attn = self.step(slots, k, v, masks)
            if i != self.iters - 1:
                slots, inds, commit = self.pool(slots)
                feedback = self.synth(slots, attn)
                v_feedback = feedback.feedback
        commits += [commit]
        pool_stats = {
            'indices': inds,
            'commit': torch.mean(torch.stack(commits), dim=0),
        }
        return slots, attn, pool_stats, feedback

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        slots = conditioning

        k_inputs = v_inputs = self.norm_input(inputs)

        if self.use_implicit_differentiation:
            slots, attn, pool_stats, alpha_weights = self.iterate(slots, k_inputs, v_inputs, masks)
            slots, attn = self.step(slots.detach(), k_inputs, v_inputs, masks)
        else:
            slots, attn, pool_stats, alpha_weights = self.iterate(slots, k_inputs, v_inputs, masks)

        stats = {
            'k_inputs': k_inputs,
            'v_inputs': v_inputs,
        }

        return slots, attn, stats, pool_stats, alpha_weights
    

class SlotAttentionInvTransScale(nn.Module):
    """Implementation of InvariantSlotAttention.
    """

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        # n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        scale_factor: int = 5,
        use_weighted_avg: bool = True,
        sg_pos_scale: bool = False,
        off_amp: bool = True
    ):
        super().__init__()
        self.dim = dim
        # self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.use_implicit_differentiation = use_implicit_differentiation

        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim

        # if self.kvq_dim % self.n_heads != 0:
        #     raise ValueError("Key, value, query dimensions must be divisible by number of heads.")
        # self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.kvq_dim**-0.5

        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        self.gru = nn.GRUCell(self.kvq_dim, dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.ff_mlp = ff_mlp

        self.grid_enc = build_two_layer_mlp(self.kvq_dim, self.kvq_dim, self.kvq_dim*2, initial_layer_norm=True)
        self.grid_proj = nn.Linear(2, self.kvq_dim, bias=True)
        self.scale_factor = scale_factor
        self.use_weighted_avg = use_weighted_avg
        self.sg_pos_scale = sg_pos_scale
        self.off_amp = off_amp

    def step(self, slots, k, v, update_slot=True, masks=None):
        slots, pos, scale = slots[..., :-4], slots[..., -4:-2], slots[..., -2:]

        for i, t in enumerate([pos, scale]):
            if torch.isnan(t).any():
                import ipdb; ipdb.set_trace()

        scale = torch.clamp(scale, min=0.001, max=2)
        pos = torch.clamp(pos, min=-1, max=1)

        bs, n_slots, _ = slots.shape
        slots_prev = slots

        with torch.cuda.amp.autocast(enabled=not self.off_amp):
            # Compute rel pos enc
            abs_grid = get_abs_grid(k.shape[1]).to(device=k.device)
            abs_grid = repeat(abs_grid, 'wh d -> b k wh d', b=bs, k=n_slots)
            rel_grid = (abs_grid - pos.unsqueeze(2)) / (scale.unsqueeze(2) * self.scale_factor)
            slots_ = self.norm_slots(slots)

            rel_enc = self.grid_proj(rel_grid)
            rel_k = self.grid_enc(k.unsqueeze(1) + rel_enc)
            rel_v = self.grid_enc(v.unsqueeze(1) + rel_enc)
            q = self.to_q(slots_).view(bs, n_slots, self.kvq_dim)
            dots = einsum(q, rel_k, "b k d, b k n d -> b k n") * self.scale
            if masks is not None:
                # Masked slots should not take part in the competition for features. By replacing their
                # dot-products with -inf, their attention values will become zero within the softmax.
                dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1), float("-inf"))
            attn = dots.softmax(dim=1)  # Take softmax over slots and heads
            attn = attn.view(bs, n_slots, -1)
            attn_before_reweighting = attn
            attn = attn + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # Compute new pos and scale
            rel_attn = attn_before_reweighting if self.use_weighted_avg else attn
            pos = einsum(rel_attn, abs_grid, 'b k n, b k n d -> b k d')
            spread = (abs_grid - pos.unsqueeze(2)).square()
            scale = torch.sqrt(
                einsum(rel_attn + self.eps, spread, 'b k n, b k n d -> b k d')
            )
            scale = torch.clamp(scale, min=0.001, max=2)

            if self.sg_pos_scale:
                pos = pos.detach()
                scale = scale.detach()

        for i, t in enumerate([pos, spread, scale, rel_grid]):
            if torch.isnan(t).any():
                import ipdb; ipdb.set_trace()

        if update_slot:
            updates = einsum(rel_v, attn, "b k n d, b k n -> b k d")
            slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))
            slots = slots.reshape(bs, -1, self.dim)

            if self.ff_mlp:
                slots = self.ff_mlp(slots)

        slots = torch.cat([slots, pos, scale], dim=-1)

        return slots, attn_before_reweighting, rel_grid

    def iterate(self, slots, k, v, masks=None):
        for i in range(self.iters + 1):
            update_slot = i < self.iters
            slots, attn, rel_grid = self.step(slots, k, v, update_slot, masks)
        return slots, attn, rel_grid

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.kvq_dim)
        v = self.to_v(inputs).view(b, n, self.kvq_dim)

        if self.use_implicit_differentiation:
            slots, attn, rel_grid = self.iterate(slots, k, v, masks)
            slots, attn, rel_grid = self.step(slots.detach(), k, v, masks)
        else:
            slots, attn, rel_grid = self.iterate(slots, k, v, masks)

        return slots, attn, rel_grid