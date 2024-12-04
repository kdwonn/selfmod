from contextlib import contextmanager
import torch
from torch import nn
from typing import Union, Callable, Optional, Tuple, Literal
import math
from einops import rearrange, repeat, einsum
import torch.nn.functional as F
from functools import partial

from ocl.conditioning import RandomConditioning
from ocl.slot_dict import FSQ, DummyQ, MemDPC, ResidualVQ, VectorQuantize
from ocl.slot_dict.feat_conditioning import DiscreteMaskedAdaGN, GroupNorm, MaskedAdaGN
import ocl.typing
from ocl.slot_dict import AdaLN, AdaLoRA, AdaScaling, ZeroModule, AdaMLP, PromptTo2D, GumbelAdaLoRA
from ocl.attention import SlotAttention


class SlotAttentionIA3(SlotAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ff_ln = nn.LayerNorm(self.dim)

    def step(self, slots, k, v, masks=None, scales: Optional[ocl.typing.ScalesIA3] = None, spatial_scale: Optional[torch.Tensor] = None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        if scales is not None:
            use_q_scale = scales.q_scale is not None
            use_v_scale = scales.v_scale is not None
            use_f_scale = scales.ff_scale is not None
        else:
            use_q_scale = False
            use_v_scale = False
            use_f_scale = False

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        if use_q_scale:
            q = q * rearrange(scales.q_scale, 'b k d -> b k 1 d')

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

        if use_v_scale:
            v = einsum(v, scales.v_scale, 'b n h d, b k d -> b k n h d')

            if spatial_scale is not None:
                v = einsum(v, spatial_scale, 'b k n h d, b k n -> b k n h d')

            updates = einsum(v, attn, 'b k n h d, b k h n -> b k h d')
        else:
            # v = repeat(v, 'b n h d -> b k n h d', k=n_slots)
            if spatial_scale is not None:
                v = einsum(v, spatial_scale, 'b n h d, b k n -> b k n h d')
                updates = einsum(v, attn, 'b k n h d, b k h n -> b k h d')
            else:
                updates = einsum(v, attn, 'b n h d, b k h n -> b k h d')

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)

        slots_ff_prev = slots

        slots = self.ff_ln(slots)
        slots = self.ff_mlp[0](slots)
        slots = self.ff_mlp[1](slots)
        if use_f_scale:
            slots = einsum(slots, scales.ff_scale, 'b k d, b k d -> b k d')
        slots = self.ff_mlp[2](slots)

        slots = slots + slots_ff_prev

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, k, v, masks=None, scales: Optional[ocl.typing.ScalesIA3] = None, spatial_scale: Optional[torch.Tensor] = None):
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, masks, scales=scales, spatial_scale=spatial_scale)
        return slots, attn

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None, scales: Optional[ocl.typing.ScalesIA3] = None, spatial_scale: Optional[torch.Tensor] = None, use_implicit_differentiation: bool = False
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks, scales=scales, spatial_scale=spatial_scale)
            slots, attn = self.step(slots.detach(), k, v, masks, scales=scales, spatial_scale=spatial_scale)
        else:
            slots, attn = self.iterate(slots, k, v, masks, scales=scales, spatial_scale=spatial_scale)

        return slots, attn


class ScalePredictor(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        normalizer, 
        use_mlp=False,
        alpha=1.0,
        learnable_alpha=False,
        beta=0.0,
    ):
        super().__init__()
        self.normalizer = normalizer
        if use_mlp:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, in_dim//4),
                nn.GELU(),
                nn.Linear(in_dim//4, out_dim)
            )
        else:
            self.proj = nn.Linear(in_dim, out_dim)

        self.alpha = alpha
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.ones(1))

        self.beta = beta

    def scale_normalization(self, scales):
        if self.normalizer == 'ln':
            return 1 + F.layer_norm(scales, (scales.shape[-1],))
        elif self.normalizer == 'sigmoid':
            return self.alpha * F.sigmoid(scales) + self.beta
        elif self.normalizer == 'tanh':
            return self.alpha * F.tanh(scales) + self.beta
        elif self.normalizer == 'none':
            return scales
        else:
            raise ValueError(f"Unknown normalizer {self.normalizer}")

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, codes):
        return self.scale_normalization(self.proj(codes))


class SlotAttentionGroupingIA3(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        use_gn=False,
        input_feature_dim=768,
        ia3_normalizer='none',
        ia3_use_mlp=False,
        ia3_acts='qvf',
        ia3_alpha=1.0,
        ia3_learnable_alpha=False,
        ia3_indim=None,
        ia3_beta=0.0,
        ia3_spatial_scale=False,
        ia3_spatial_centering=True,
    ):
        super().__init__()
        self._object_dim = object_dim
        self.slot_attention = SlotAttentionIA3(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
        )

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

        self.use_gn = use_gn
        if self.use_gn:
            self.gn = nn.GroupNorm(num_groups=24, num_channels=input_feature_dim)

        ia3_indim = object_dim if ia3_indim is None else ia3_indim
        self.q_scale_pred = ScalePredictor(
            ia3_indim, 
            self.slot_attention.kvq_dim, 
            normalizer=ia3_normalizer, 
            use_mlp=ia3_use_mlp,
            alpha=ia3_alpha,
            learnable_alpha=ia3_learnable_alpha,
            beta=ia3_beta,
        ) if 'q' in ia3_acts else None
        self.v_scale_pred = ScalePredictor(
            ia3_indim, 
            self.slot_attention.kvq_dim, 
            normalizer=ia3_normalizer, 
            use_mlp=ia3_use_mlp,
            alpha=ia3_alpha,
            learnable_alpha=ia3_learnable_alpha,
            beta=ia3_beta,
        ) if 'v' in ia3_acts else None
        self.ff_scale_pred = ScalePredictor(
            ia3_indim, 
            object_dim*4, 
            normalizer=ia3_normalizer, 
            use_mlp=ia3_use_mlp,
            alpha=ia3_alpha,
            learnable_alpha=ia3_learnable_alpha,
            beta=ia3_beta,
        ) if 'f' in ia3_acts else None

        self.ia3_spatial_scale = ia3_spatial_scale
        self.ia3_spatial_centering = ia3_spatial_centering

    def scale_default(self, scale_pred, code):
        if scale_pred is not None:
            return scale_pred(code)
        return None

    @property
    def object_dim(self):
        return self._object_dim

    def spatial_scale_centering(self, spatial_scale):
        if spatial_scale is not None:
            if self.ia3_spatial_centering:
                return 1 + (spatial_scale - spatial_scale.mean(dim=-1, keepdim=True))
            return spatial_scale
        return None

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
        code: Optional[torch.Tensor] = None,
        pre_attn: Optional[torch.Tensor] = None,
        use_implicit_differentiation: bool = False,
    ) -> ocl.typing.PerceptualGroupingOutput:
        if self.use_gn:
            feature.features = self.gn(feature.features.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_embedding:
            feature = self.positional_embedding(feature.features, feature.positions)
        else:
            feature = feature.features

        scales = None
        if code is not None:
            scales = ocl.typing.ScalesIA3(
                q_scale=self.scale_default(self.q_scale_pred, code),
                v_scale=self.scale_default(self.v_scale_pred, code),
                ff_scale=self.scale_default(self.ff_scale_pred, code),
            )

        pre_attn = self.spatial_scale_centering(pre_attn)
        slots, attn = self.slot_attention(feature, conditioning, slot_mask, scales, pre_attn if self.ia3_spatial_scale else None, use_implicit_differentiation=use_implicit_differentiation)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingOutput(
            slots, feature_attributions=attn, is_empty=slot_mask, scales=scales
        )


class SlotAttentionIA3Iterwise(SlotAttention):
    def __init__(
        self, 
        *args, 
        ia3_normalizer='none',
        ia3_use_mlp=False,
        ia3_acts='qvf',
        ia3_alpha=1.0,
        ia3_learnable_alpha=False,
        ia3_beta=0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ff_ln = nn.LayerNorm(self.dim)

        self.q_scale_pred = ScalePredictor(
            self.dim, 
            self.kvq_dim, 
            normalizer=ia3_normalizer, 
            use_mlp=ia3_use_mlp,
            alpha=ia3_alpha,
            learnable_alpha=ia3_learnable_alpha,
            beta=ia3_beta,
        ) if 'q' in ia3_acts else None
        self.v_scale_pred = ScalePredictor(
            self.dim, 
            self.kvq_dim, 
            normalizer=ia3_normalizer, 
            use_mlp=ia3_use_mlp,
            alpha=ia3_alpha,
            learnable_alpha=ia3_learnable_alpha,
            beta=ia3_beta,
        ) if 'v' in ia3_acts else None
        self.ff_scale_pred = ScalePredictor(
            self.dim, 
            self.dim*4, 
            normalizer=ia3_normalizer, 
            use_mlp=ia3_use_mlp,
            alpha=ia3_alpha,
            learnable_alpha=ia3_learnable_alpha,
            beta=ia3_beta,
        ) if 'f' in ia3_acts else None

    def scales_default(self, slots):
        # None if scale pred is None
        def scale_default(scale_pred, code):
            if scale_pred is not None:
                return scale_pred(code)
            return None
        return ocl.typing.ScalesIA3(
            q_scale=scale_default(self.q_scale_pred, slots),
            v_scale=scale_default(self.v_scale_pred, slots),
            ff_scale=scale_default(self.ff_scale_pred, slots),
        )

    def step(self, slots, k, v, masks=None, scales: Optional[ocl.typing.ScalesIA3] = None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        if scales is not None:
            use_q_scale = scales.q_scale is not None
            use_v_scale = scales.v_scale is not None
            use_f_scale = scales.ff_scale is not None
        else:
            use_q_scale = False
            use_v_scale = False
            use_f_scale = False

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        if use_q_scale:
            q = q * rearrange(scales.q_scale, 'b k d -> b k 1 d')

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

        if use_v_scale:
            v = einsum(v, scales.v_scale, 'b n h d, b k d -> b k n h d')

            updates = einsum(v, attn, 'b k n h d, b k h n -> b k h d')
        else:
            updates = einsum(v, attn, 'b n h d, b k h n -> b k h d')

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)

        slots_ff_prev = slots

        slots = self.ff_ln(slots)
        slots = self.ff_mlp[0](slots)
        slots = self.ff_mlp[1](slots)
        if use_f_scale:
            slots = einsum(slots, scales.ff_scale, 'b k d, b k d -> b k d')
        slots = self.ff_mlp[2](slots)

        slots = slots + slots_ff_prev

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, k, v, masks=None):
        for _ in range(self.iters):
            scales = self.scales_default(slots)
            slots, attn = self.step(slots, k, v, masks, scales=scales)
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
            scales = self.scales_default(slots)
            slots, attn = self.step(slots.detach(), k, v, masks, scales=scales)
        else:
            slots, attn = self.iterate(slots, k, v, masks)

        return slots, attn


class SlotAttentionGroupingIA3Iterwise(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        use_gn=False,
        input_feature_dim=768,
        ia3_normalizer='none',
        ia3_use_mlp=False,
        ia3_acts='qvf',
        ia3_alpha=1.0,
        ia3_learnable_alpha=False,
        ia3_beta=0.0,
    ):
        super().__init__()
        self._object_dim = object_dim
        self.slot_attention = SlotAttentionIA3Iterwise(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
            ia3_normalizer=ia3_normalizer,
            ia3_use_mlp=ia3_use_mlp,
            ia3_acts=ia3_acts,
            ia3_alpha=ia3_alpha,
            ia3_learnable_alpha=ia3_learnable_alpha,
            ia3_beta=ia3_beta,
        )

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

        self.use_gn = use_gn
        if self.use_gn:
            self.gn = nn.GroupNorm(num_groups=24, num_channels=input_feature_dim)

    def scale_default(self, scale_pred, code):
        if scale_pred is not None:
            return scale_pred(code)
        return None

    @property
    def object_dim(self):
        return self._object_dim

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        if self.use_gn:
            feature.features = self.gn(feature.features.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_embedding:
            feature = self.positional_embedding(feature.features, feature.positions)
        else:
            feature = feature.features

        slots, attn = self.slot_attention(feature, conditioning, slot_mask)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingOutput(
            slots, feature_attributions=attn, is_empty=slot_mask
        )


class SlotAttentionMod(nn.Module):
    def __init__(
        self,
        dim: int,
        feature_dim: int,
        pool: Union[VectorQuantize, ResidualVQ, FSQ, DummyQ, MemDPC],
        reduction: bool,
        rank: int = 8,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        mod_position: str = 'f',
        use_gumbel: bool = False,
        lora_scale: int = 8,
        gumbel_temperature: int = 1
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
        self.ff_mlp = ff_mlp

        self.pool = pool
        self.num_entries = pool.codebook_size
        self.mod_position = mod_position

        self.norm_slots = nn.LayerNorm(dim)
        self.norm_slots_ff_mlp = nn.LayerNorm(dim)

        self.mod_in_k = 'k' in mod_position
        self.mod_in_v = 'v' in mod_position
        self.mod_in_q = 'q' in mod_position
        self.mod_in_f = 'f' in mod_position
        self.mod_in_F = 'F' in mod_position

        self.gumbel_temperature = gumbel_temperature
        self.use_gumbel = use_gumbel
        LoRA = GumbelAdaLoRA if use_gumbel else AdaLoRA

        self.k_lora = (
            LoRA(dim, rank, self.num_entries, scale=lora_scale, reduction=True) 
            if self.mod_in_k 
            else ZeroModule()
        )
        self.v_lora = (
            LoRA(dim, rank, self.num_entries, scale=lora_scale, reduction=True) 
            if self.mod_in_v
            else ZeroModule()
        )
        self.q_lora = (
            LoRA(dim, rank, self.num_entries, scale=lora_scale, reduction=reduction)
            if self.mod_in_q
            else ZeroModule()
        )
        self.f_lora1 = (
            LoRA(dim, rank, self.num_entries, scale=lora_scale, reduction=reduction, out_dim=4*dim)
            if self.mod_in_f
            else ZeroModule()
        )
        self.f_lora2 = (
            LoRA(4*dim, rank, self.num_entries, scale=lora_scale, reduction=reduction, out_dim=dim)
            if self.mod_in_f
            else ZeroModule()
        )
        self.ff_adamlp = (
            AdaMLP(dim, 4, self.num_entries, reduction=reduction)
            if self.mod_in_F
            else None
        )

        self.zero_module = ZeroModule()

    def cond_lora(self, lora, x, indices):
        if indices is None:
            return self.zero_module(x)
        else:
            if self.use_gumbel:
                lora = partial(lora, temperature=self.gumbel_temperature)
            return lora(x, indices)
        
    def cond_lora_mlp(self, x, indices):
        if indices is None or not self.mod_in_f:
            return self.ff_mlp(x)
        else:
            assert len(self.ff_mlp) == 3
            if self.use_gumbel:
                lora1 = partial(self.f_lora1, temperature=self.gumbel_temperature)
                lora2 = partial(self.f_lora2, temperature=self.gumbel_temperature)
            x = self.ff_mlp[0](x) + lora1(x, indices)
            x = self.ff_mlp[1](x)
            x = self.ff_mlp[2](x) + lora2(x, indices)
            return x
        
    def cond_mlp(self, mlp, x, indices):
        if indices is None or not self.mod_in_F:
            return self.ff_mlp(x)
        else:
            return mlp(x, indices)
        
    def get_logit(self, slots):
        if self.use_gumbel:
            _, indices, commit, logits = self.pool(slots)
        else:
            _, indices, commit = self.pool(slots)
            logits = indices
        return indices, commit, logits

    def step(self, slots, k, v, indices, masks=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        q = (q + self.cond_lora(self.q_lora, slots, indices)).view(
            bs, n_slots, self.n_heads, self.dims_per_head
        )

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)
        slots_prev_ff_mlp = slots
        slots = self.norm_slots_ff_mlp(slots)

        # slots_ff = self.cond_mlp(self.ff_adamlp, slots, indices)
        slots = self.cond_lora_mlp(slots, indices)
        
        slots = slots + slots_prev_ff_mlp

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, inputs, masks=None):
        b, n, d = inputs.shape
        for i in range(self.iters):
            if i == 0:
                logit = None
            else:
                indices, commit, logit = self.get_logit(slots)
            
            k = self.to_k(inputs)
            k = (k + self.cond_lora(self.k_lora, inputs, logit)).view(
                b, n, self.n_heads, self.dims_per_head
            )
            v = self.to_v(inputs)
            v = (v + self.cond_lora(self.v_lora, inputs, logit)).view(
                b, n, self.n_heads, self.dims_per_head
            )
            
            slots, attn = self.step(slots, k, v, logit, masks)
        return slots, attn, indices, commit, logit

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning
        inputs = self.norm_input(inputs)

        if self.use_implicit_differentiation:
            slots, attn, indices, commit, logit = self.iterate(slots, inputs, masks)
            indices, commit, logit = self.get_logit(slots)
            k = self.to_k(inputs)
            k = (k + self.cond_lora(self.k_lora, inputs, logit)).view(
                b, n, self.n_heads, self.dims_per_head
            )
            v = self.to_v(inputs)
            v = (v + self.cond_lora(self.v_lora, inputs, logit)).view(
                b, n, self.n_heads, self.dims_per_head
            )
            slots, attn = self.step(slots.detach(), k, v, logit, masks)
        else:
            slots, attn, indices, commit, logit = self.iterate(slots, inputs, masks)

        return slots, attn, indices, commit, logit


class SlotAttentionGroupingMod(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        pool: nn.Module,
        mod_position,
        reduction: bool,
        rank: int = 8,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        use_gumbel: bool = False,
        lora_scale: int = 8,
        gumbel_temperature: int = 1
    ):
        super().__init__()
        self._object_dim = object_dim
        self.slot_attention = SlotAttentionMod(
            dim=object_dim,
            feature_dim=feature_dim,
            pool=pool,
            mod_position=mod_position,
            reduction=reduction,
            rank=rank,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
            use_gumbel=use_gumbel,
            lora_scale=lora_scale,
            gumbel_temperature=gumbel_temperature
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

        slots, attn, indices, commit, logit = self.slot_attention(feature, conditioning, slot_mask)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingOutput(
            slots, 
            feature_attributions=attn, 
            is_empty=slot_mask, 
            pool_indices=indices, 
            commit_loss=commit.sum(),
            pool_logit=logit,
            gumbel_temp=self.slot_attention.gumbel_temperature
        )


class SlotAttentionModExt(nn.Module):
    def __init__(
        self,
        dim: int,
        feature_dim: int,
        reduction: bool,
        num_entries: int,
        rank: int = 8,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        mod_position: str = 'f',
        use_gumbel: bool = False,
        lora_scale: int = 8,
        use_o: bool = True,
        gumbel_temperature: int = 1
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

        self.use_o = use_o
        if self.use_o:
            self.to_o = nn.Linear(self.kvq_dim, dim, bias=use_projection_bias)
        else:
            self.gru = nn.GRUCell(self.kvq_dim, dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.ff_mlp = ff_mlp

        self.num_entries = num_entries

        self.norm_slots = nn.LayerNorm(dim)
        self.norm_slots_ff_mlp = nn.LayerNorm(dim)

        self.mod_in_k = 'k' in mod_position
        self.mod_in_v = 'v' in mod_position
        self.mod_in_q = 'q' in mod_position
        self.mod_in_f = 'f' in mod_position
        self.mod_in_F = 'F' in mod_position
        self.mod_in_o = 'o' in mod_position

        self.use_gumbel = use_gumbel
        self.gumbel_temperature = gumbel_temperature
        LoRA = GumbelAdaLoRA if use_gumbel else AdaLoRA

        self.k_lora = (
            LoRA(dim, rank, self.num_entries, scale=lora_scale, reduction=True) 
            if self.mod_in_k 
            else ZeroModule()
        )
        self.v_lora = (
            LoRA(dim, rank, self.num_entries, scale=lora_scale, reduction=True) 
            if self.mod_in_v
            else ZeroModule()
        )
        self.q_lora = (
            LoRA(dim, rank, self.num_entries, scale=lora_scale, reduction=reduction)
            if self.mod_in_q
            else ZeroModule()
        )
        self.o_lora = (
            LoRA(dim, rank, self.num_entries, scale=lora_scale, reduction=reduction)
            if self.mod_in_o
            else ZeroModule()
        )
        self.f_lora1 = (
            LoRA(dim, rank, self.num_entries, scale=lora_scale, reduction=reduction, out_dim=4*dim)
            if self.mod_in_f
            else ZeroModule()
        )
        self.f_lora2 = (
            LoRA(4*dim, rank, self.num_entries, scale=lora_scale, reduction=reduction, out_dim=dim)
            if self.mod_in_f
            else ZeroModule()
        )
        self.ff_adamlp = (
            AdaMLP(dim, 4, self.num_entries, reduction=reduction)
            if self.mod_in_F
            else None
        )

        self.zero_module = ZeroModule()

    def cond_lora(self, lora, x, indices):
        if indices is None:
            return self.zero_module(x)
        else:
            if self.use_gumbel:
                lora = partial(lora, temperature=self.gumbel_temperature)
            return lora(x, indices)
        
    def cond_lora_mlp(self, x, indices):
        if indices is None or not self.mod_in_f:
            return self.ff_mlp(x)
        else:
            assert len(self.ff_mlp) == 3
            if self.use_gumbel:
                lora1 = partial(self.f_lora1, temperature=self.gumbel_temperature)
                lora2 = partial(self.f_lora2, temperature=self.gumbel_temperature)
            x = self.ff_mlp[0](x) + lora1(x, indices)
            x = self.ff_mlp[1](x)
            x = self.ff_mlp[2](x) + lora2(x, indices)
            return x
        
    def cond_mlp(self, mlp, x, indices):
        if indices is None or not self.mod_in_F:
            return self.ff_mlp(x)
        else:
            return mlp(x, indices)

    def step(self, slots, k, v, indices=None, masks=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        q = (q + self.cond_lora(self.q_lora, slots, indices)).view(
            bs, n_slots, self.n_heads, self.dims_per_head
        )

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        if self.use_o:
            updates = rearrange(updates, 'b k h d -> b k (h d)')
            slots = self.to_o(updates) + self.cond_lora(self.o_lora, slots, indices)
            slots = slots + slots_prev
        else:
            slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)
        slots_prev_ff_mlp = slots
        slots = self.norm_slots_ff_mlp(slots)

        # slots_ff = self.cond_mlp(self.ff_adamlp, slots, indices)
        slots = self.cond_lora_mlp(slots, indices)
        
        slots = slots + slots_prev_ff_mlp

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, k, v, indices=None, masks=None):
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, indices, masks)
        return slots, attn, indices

    def forward(
        self,
        inputs: torch.Tensor,
        conditioning: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        k = (k + self.cond_lora(self.k_lora, inputs, indices)).view(
            b, n, self.n_heads, self.dims_per_head
        )
        v = self.to_v(inputs)
        v = (v + self.cond_lora(self.v_lora, inputs, indices)).view(
            b, n, self.n_heads, self.dims_per_head
        )

        if self.use_implicit_differentiation:
            slots, attn, indices = self.iterate(slots, k, v, indices, masks)
            slots, attn = self.step(slots.detach(), k, v, indices, masks)
        else:
            slots, attn, indices = self.iterate(slots, k, v, indices, masks)

        return slots, attn, indices
    

class SlotAttentionGroupingModExt(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        pool: nn.Module,
        reduction: bool,
        rank: int = 8,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        mod_position: str = 'f',
        use_gumbel: bool = False,
        lora_scale: int = 8,
        use_o: bool = False,
        gumbel_temperature: int = 1
    ):
        super().__init__()
        self._object_dim = object_dim
        self.slot_attention = SlotAttentionModExt(
            dim=object_dim,
            feature_dim=feature_dim,
            reduction=reduction,
            num_entries=pool.codebook_size,
            rank=rank,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
            mod_position=mod_position,
            use_gumbel=use_gumbel,
            lora_scale=lora_scale,
            use_o=use_o,
            gumbel_temperature=gumbel_temperature
        )

        self.use_gumbel = use_gumbel

        self.positional_embedding = positional_embedding
        self.pool = pool
        self.feature_dim = feature_dim

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
        if self.positional_embedding:
            feature = self.positional_embedding(feature.features, feature.positions)
        else:
            feature = feature.features

        pre_slots, pre_attn, _ = self.slot_attention(feature, conditioning, masks=slot_mask)
        _, indices, commit = self.pool(pre_slots)
        slots, attn, _ = self.slot_attention(feature, conditioning, indices=indices, masks=slot_mask)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingOutput(
            slots, 
            feature_attributions=attn, 
            is_empty=slot_mask, 
            pool_indices=indices, 
            commit_loss=commit.sum(),
            pre_slots=pre_slots,
            pre_attn=pre_attn,
        )
    

class SlotAttentionGroupingModExtDec(SlotAttentionGroupingModExt):
    def __init__(self, object_decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = object_decoder

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_except_mod(self):
        for param in self.parameters():
            param.requires_grad = False
        for module in self.slot_attention.modules():
            if isinstance(module, (AdaLoRA, AdaMLP, GumbelAdaLoRA)):
                for param in module.parameters():
                    param.requires_grad = True

    def freeze_mod(self):
        for module in self.slot_attention.modules():
            if isinstance(module, (AdaLoRA, AdaMLP, GumbelAdaLoRA)):
                for param in module.parameters():
                    param.requires_grad = False
    
    def get_logit(self, slots):
        if self.use_gumbel:
            _, indices, commit, logits = self.pool(slots)
        else:
            _, indices, commit = self.pool(slots)
            logits = indices
        return indices, commit, logits

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        train_base: bool,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
        target: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        
        self.unfreeze()
        if train_base:
            self.freeze_mod()
        else:
            self.freeze_except_mod()

        if self.positional_embedding:
            feature = self.positional_embedding(feature.features, feature.positions)
        else:
            feature = feature.features

        #  first pass
        pre_slots, pre_attn, _ = self.slot_attention(feature, conditioning, masks=slot_mask)
        indices, commit, logits = self.get_logit(pre_slots)
        
        #  second pass
        slots, attn, _ = self.slot_attention(
            feature.detach(), conditioning.detach(), indices=logits, masks=slot_mask)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        if self.training:
            if train_base:
                post_decoding = None
                pre_decoding = self.decoder(pre_slots, target, image=image)
                dec_to_train = pre_decoding
            else:
                post_decoding = self.decoder(slots, target, image=image)
                pre_decoding = None
                dec_to_train = post_decoding
        else:
            post_decoding = self.decoder(slots, target, image=image)
            pre_decoding = self.decoder(pre_slots, target, image=image)
            dec_to_train = post_decoding

        # with torch.no_grad():
        #     print(f'only train base?: {train_base}')
        #     print(f'lora weight:{self.slot_attention.q_lora.up_proj_values[0,0,:3]}')
        #     print(f'base sa weight: {self.slot_attention.to_q.weight[0,:3]}')
        #     print()

        return ocl.typing.PerceptualGroupingModDecOutput(
            slots,
            dec_to_train,
            pre_object_decoder=pre_decoding,
            object_decoder=post_decoding,
            feature_attributions=attn, 
            is_empty=slot_mask, 
            pool_indices=indices, 
            commit_loss=commit.sum(),
            pre_slots=pre_slots,
            pre_attn=pre_attn,
            pool_logit = logits,
            gumbel_temp=self.slot_attention.gumbel_temperature
        )
    

class SlotAttentionGroupingModAltAttn(SlotAttentionGroupingModExtDec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def freeze_except_mod(self):
        for p in self.slot_attention.parameters():
            p.requires_grad = False
        for module in self.slot_attention.modules():
            if isinstance(module, (AdaLoRA, AdaMLP, GumbelAdaLoRA)):
                for p in module.parameters():
                    p.requires_grad = True

    def freeze_mod(self):
        for module in self.slot_attention.modules():
            if isinstance(module, (AdaLoRA, AdaMLP, GumbelAdaLoRA)):
                for param in module.parameters():
                    param.requires_grad = False

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        train_base: bool,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
        target: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        
        self.unfreeze()
        if train_base:
            self.freeze_mod()
        else:
            self.freeze_except_mod()

        if self.positional_embedding:
            feature = self.positional_embedding(feature.features, feature.positions)
        else:
            feature = feature.features

        #  first pass
        pre_slots, pre_attn, _ = self.slot_attention(feature, conditioning, masks=slot_mask)
        indices, commit, logits = self.get_logit(pre_slots)
        
        #  second pass
        slots, attn, _ = self.slot_attention(
            feature, conditioning, indices=logits, masks=slot_mask)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        if self.training:
            if train_base:
                post_decoding = None
                pre_decoding = self.decoder(pre_slots, target, image=image)
                dec_to_train = pre_decoding
            else:
                post_decoding = self.decoder(slots, target, image=image)
                pre_decoding = None
                dec_to_train = post_decoding
        else:
            post_decoding = self.decoder(slots, target, image=image)
            pre_decoding = self.decoder(pre_slots, target, image=image)
            dec_to_train = post_decoding

        # with torch.no_grad():
        #     print(f'only train base?: {train_base}')
        #     print(f'lora weight:{self.slot_attention.q_lora.up_proj_values[0,0,:3]}')
        #     print(f'base sa weight: {self.slot_attention.to_q.weight[0,:3]}')
        #     print()
        #     import ipdb; ipdb.set_trace()

        return ocl.typing.PerceptualGroupingModDecOutput(
            slots,
            dec_to_train,
            pre_object_decoder=pre_decoding,
            object_decoder=post_decoding,
            feature_attributions=attn, 
            is_empty=slot_mask, 
            pool_indices=indices, 
            commit_loss=commit.sum(),
            pre_slots=pre_slots,
            pre_attn=pre_attn,
            pool_logit = logits,
            gumbel_temp=self.slot_attention.gumbel_temperature
        )


class SlotAttentionGroupingPromptExtDec(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        object_decoder: nn.Module,
        base_update_freq: int,
        base_warmup_iter: int,
        pool: nn.Module,
        rank: int = 8,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        prompt_weight: int = 0.5,
        use_soft_prompt: bool = False,
    ):
        super().__init__()
        self.decoder = object_decoder
        self.iters = 0
        self.freq = base_update_freq
        self.warmup = base_warmup_iter
        self.positional_embedding = positional_embedding
        self.pool = pool
        self.feature_dim = feature_dim

        self.prompt_dict = PromptTo2D(dim=self.feature_dim, num_entries=self.pool.codebook_size, use_soft=use_soft_prompt)
        self.null_prompt = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        nn.init.xavier_normal_(self.null_prompt)
        self.prompt_weight = prompt_weight
        
        self._object_dim = object_dim
        self.slot_attention = SlotAttention(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
        )

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

    @property
    def object_dim(self):
        return self._object_dim

    @contextmanager
    def only_train_prompt(self, cond):
        try:
            for param in self.parameters():
                param.requires_grad = False
            if cond:
                for param in self.prompt_dict.parameters():
                    param.requires_grad = True
            yield
        finally:
            # Revert requires_grad settings to their original state
            for param in self.parameters():
                param.requires_grad = True

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
        target: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        if self.iters >= self.warmup:
            train_prompt = self.iters % self.freq != 0
        else:
            train_prompt = False

        if self.positional_embedding:
            feature = self.positional_embedding(feature.features, feature.positions)
        else:
            feature = feature.features

        pre_slots, pre_attn = self.slot_attention(feature + self.prompt_weight*self.null_prompt, conditioning, masks=slot_mask)
        _, indices, commit = self.pool(pre_slots)
        pre_decoding = self.decoder(pre_slots, target, image=image)
        
        with self.only_train_prompt(train_prompt):
            prompt = self.prompt_dict(indices, pre_attn.detach())
            prompted_feature = feature + self.prompt_weight*prompt
            slots, attn = self.slot_attention(prompted_feature, conditioning, masks=slot_mask)
            post_decoding = self.decoder(slots, target, image=image)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        dec_to_train = post_decoding if train_prompt else pre_decoding

        self.iters += 1

        return ocl.typing.PerceptualGroupingModDecOutput(
            slots,
            dec_to_train,
            prompt=prompt,
            prompted_feature=prompted_feature,
            feature=feature,
            pre_object_decoder=pre_decoding,
            object_decoder=post_decoding,
            feature_attributions=attn, 
            is_empty=slot_mask, 
            pool_indices=indices, 
            commit_loss=commit.sum(),
            pre_slots=pre_slots,
            pre_attn=pre_attn,
        )


class SlotAttentionGroupingAdaGN(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        input_feature_dim: int,
        pool: nn.Module,
        positional_embedding: nn.Module,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        pre_post_sharing: bool = True,
        pre_gn: bool = False,
        discrete_gn: bool= False,
        condition_sharing: bool = False,
        num_slots: int = 7,
        adagn_pre_ln: bool = True,
    ):
        super().__init__()
        self._object_dim = object_dim

        get_slot_attn = lambda: SlotAttention(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
        )

        self.slot_attention = get_slot_attn()

        if pre_post_sharing:
            self.pre_slot_attention = self.slot_attention
        else:
            self.pre_slot_attention = get_slot_attn()

        self.positional_embedding = positional_embedding
        self.pool = pool
        self.feature_dim = feature_dim

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None
        
        self.discrete_gn = discrete_gn
        if discrete_gn:
            self.masked_adagn = DiscreteMaskedAdaGN(object_dim, input_feature_dim, pool.codebook_size)
        else:
            self.masked_adagn = MaskedAdaGN(object_dim, input_feature_dim, pre_ln=adagn_pre_ln)
        assert not discrete_gn or pre_post_sharing # discrete_gn -> pre_post_sharing

        self.condition_sharing = condition_sharing
        if not self.condition_sharing:
            self.pre_conditioning = RandomConditioning(
                object_dim=object_dim,
                n_slots=num_slots,
            )

        assert pre_gn ^ discrete_gn # only one of the two can be true
        self.pre_gn = GroupNorm(input_feature_dim) if pre_gn else self.masked_adagn

    @property
    def object_dim(self):
        return self._object_dim

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        
        pos_feat = self.positional_embedding(
            self.pre_gn(feature.features), 
            feature.positions
        )
        if self.condition_sharing:
            pre_conditioning = conditioning
        else:
            pre_conditioning = self.pre_conditioning(conditioning.shape[0])
        pre_slots, pre_attn = self.pre_slot_attention(pos_feat, pre_conditioning, masks=slot_mask)

        q_slots, indices, commit = self.pool(pre_slots)
        
        if self.discrete_gn:
            ada_feature = self.masked_adagn(feature.features, indices, pre_attn)
        else:
            ada_feature = self.masked_adagn(feature.features, q_slots, pre_attn)
        pos_ada_feat = self.positional_embedding(ada_feature, feature.positions)
        slots, attn = self.slot_attention(pos_ada_feat, conditioning, masks=slot_mask)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingAdaGNOutput(
            slots, 
            feature_attributions=attn, 
            is_empty=slot_mask, 
            pool_indices=indices, 
            commit_loss=commit.sum(),
            pre_slots=pre_slots,
            pre_attn=pre_attn,
            feature=feature.features,
            ada_feature=ada_feature,
        )
    

class SlotAttentionGroupingAdaGNDec(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        input_feature_dim: int,
        pool: nn.Module,
        positional_embedding: nn.Module,
        object_decoder: nn.Module,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        pre_post_sharing: bool = True,
        pre_gn: bool = False,
        discrete_gn: bool= False,
    ):
        super().__init__()
        self._object_dim = object_dim

        get_slot_attn = lambda: SlotAttention(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
        )

        self.slot_attention = get_slot_attn()

        if pre_post_sharing:
            self.pre_slot_attention = self.slot_attention
        else:
            self.pre_slot_attention = get_slot_attn()

        self.positional_embedding = positional_embedding
        self.pool = pool
        self.feature_dim = feature_dim

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None
        
        self.discrete_gn = discrete_gn
        if discrete_gn:
            self.masked_adagn = DiscreteMaskedAdaGN(object_dim, input_feature_dim, pool.codebook_size)
        else:
            self.masked_adagn = MaskedAdaGN(object_dim, input_feature_dim)
        assert not discrete_gn or pre_post_sharing

        self.pre_gn = nn.GroupNorm(24, input_feature_dim) if pre_gn else nn.Identity()

        self.object_decoder = object_decoder

    @property
    def object_dim(self):
        return self._object_dim

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
        target: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        
        pos_feat = self.positional_embedding(
            self.pre_gn(feature.features.permute(0, 2, 1)).permute(0, 2, 1), 
            feature.positions
        )
        pre_slots, pre_attn = self.pre_slot_attention(pos_feat, conditioning, masks=slot_mask)

        q_slots, indices, commit = self.pool(pre_slots)

        pre_decoding = self.object_decoder(q_slots, target, image=image)
        
        if self.discrete_gn:
            ada_feature = self.masked_adagn(feature.features, indices, pre_attn)
        else:
            ada_feature = self.masked_adagn(feature.features, q_slots, pre_attn)
        pos_ada_feat = self.positional_embedding(ada_feature, feature.positions)
        slots, attn = self.slot_attention(pos_ada_feat, conditioning, masks=slot_mask)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        decoding = self.object_decoder(slots, target, image=image)

        return ocl.typing.PerceptualGroupingAdaGNDecOutput(
            slots, 
            decoding,
            pre_decoding,
            feature_attributions=attn, 
            is_empty=slot_mask, 
            pool_indices=indices, 
            commit_loss=commit.sum(),
            pre_slots=pre_slots,
            pre_attn=pre_attn,
            feature=feature.features,
            ada_feature=ada_feature,
        )