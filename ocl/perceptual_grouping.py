"""Implementations of perceptual grouping algorithms.

We denote methods that group input feature together into slots of objects
(either unconditionally) or via additional conditioning signals as perceptual
grouping modules.
"""
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
from ocl.slot_dict import VectorQuantize, ResidualVQ, FSQ, DummyQ, MemDPC
from ocl.slot_dict import BlockGRU, BlockLinear, BlockLayerNorm
from ocl.synthesizer import BraodcastingSynthesizer, MaskedBroadcastingSynthesizer, SynthPrompt, DivergenceSynthesizer, SoftMaskedBroadcastingSynthesizer
from ocl.decoding import PatchDecoder, AutoregressivePatchDecoder
from ocl.neural_networks import build_two_layer_mlp, Sequential, DummyPositionEmbed
from ocl.conditioning import RandomConditioning
from ocl.attention import *
from ocl.hypernet import Hypernet
from ocl.abs_modules import *


def detach(x: torch.Tensor, condition: bool = True) -> torch.Tensor:
    """Detach tensor."""
    return x.detach() if condition else x


class SlotAttentionGrouping(nn.Module):
    """Implementation of SlotAttention for perceptual grouping."""

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
    ):
        """Initialize Slot Attention Grouping.

        Args:
            feature_dim: Dimensionality of features to slot attention (after positional encoding).
            object_dim: Dimensionality of slots.
            kvq_dim: Dimensionality after projecting to keys, values, and queries. If `None`,
                `object_dim` is used.
            n_heads: Number of heads slot attention uses.
            iters: Number of slot attention iterations.
            eps: Epsilon in slot attention.
            ff_mlp: Optional module applied slot-wise after GRU update.
            positional_embedding: Optional module applied to the features before slot attention,
                adding positional encoding.
            use_projection_bias: Whether to use biases in key, value, query projections.
            use_implicit_differentiation: Whether to use implicit differentiation trick. If true,
                performs one more iteration of slot attention that is used for the gradient step
                after `iters` iterations of slot attention without gradients. Faster and more memory
                efficient than the standard version, but can not backpropagate gradients to the
                conditioning input.
            use_empty_slot_for_masked_slots: Replace slots masked with a learnt empty slot vector.
        """
        super().__init__()
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

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

        self.use_gn = use_gn
        if self.use_gn:
            self.gn = nn.GroupNorm(num_groups=24, num_channels=input_feature_dim)

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


class SlotAttentionGroupingAttnMod(nn.Module):

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
        mod_weight=0.5,
    ):
        super().__init__()
        self._object_dim = object_dim
        self.slot_attention = SlotAttentionAttnMod(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
            mod_weight=mod_weight,
        )

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

        self.use_gn = use_gn
        if self.use_gn:
            self.gn = nn.GroupNorm(num_groups=24, num_channels=input_feature_dim)

    @property
    def object_dim(self):
        return self._object_dim

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
        attn_mod: Optional[torch.Tensor] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        """Apply slot attention based perceptual grouping.

        Args:
            feature: Features used for grouping.
            conditioning: Initial conditioning vectors for slots.
            slot_mask: Slot mask where true indicates that the slot should be masked.

        Returns:
            The grouped features.
        """
        if self.use_gn:
            feature.features = self.gn(feature.features.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_embedding:
            feature = self.positional_embedding(feature.features, feature.positions)
        else:
            feature = feature.features

        slots, attn = self.slot_attention(feature, conditioning, slot_mask, attn_mod=attn_mod)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingOutput(
            slots, feature_attributions=attn, is_empty=slot_mask, mod_weight=self.slot_attention.mod_weight
        )


class SlotAttentionGroupingABS(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        pool: Union[VectorQuantize, ResidualVQ, FSQ],
        feat_num: int,
        feedback_type: str,
        synthesizer: Union[BraodcastingSynthesizer, MaskedBroadcastingSynthesizer],
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        synth_feat_without_pos: bool = False,
        use_synth_prompt: bool = False,
        synth_in_detach: bool = False,
        num_blocks: int = 1,
        feedback_scale: float = 1.0,
        feedback_gating: bool = False,
        scale_sum_to_one: bool = False,
        feedback_normalization: bool = False,
        var_synth_in_detach: bool = False,
        td_downsample: int = 1,
        prompting_feedback: bool = False,
    ):
        """Initialize Slot Attention Grouping.

        Args:
            pool: slot pool to use: choose among VQ, RVQ, FSQ
            pool_config: slot pool configuration
            **kwargs: other arguments for SlotAttentionGrouping
        """
        super(SlotAttentionGroupingABS, self).__init__()
        self._object_dim = object_dim
        slotwise_feedback = getattr(synthesizer, 'slotwise', False) 
        self.slot_attention = SlotAttentionWithFeedback(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
            num_blocks=num_blocks,
            feedback_scale=feedback_scale,
            feedback_gating=feedback_gating,
            scale_sum_to_one=scale_sum_to_one,
            feedback_normalization=feedback_normalization,
            feedback_slotwise=slotwise_feedback,
        )

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None
        
        self.slot_pool = pool
        self.feedback_type = feedback_type
        self.synth = synthesizer
        self.synth_feat_without_pos = synth_feat_without_pos

        self.use_synth_prompt = use_synth_prompt
        if self.use_synth_prompt:
            self.synth_prompt = SynthPrompt(feature_dim, feat_num, feedback_type)

        self.synth_in_detach = synth_in_detach
        self.var_synth_in_detach = var_synth_in_detach

        self.td_downsample = td_downsample
        self.downsampler = None
        if self.td_downsample > 1:
            self.downsampler = nn.Conv2d(feature_dim, feature_dim, kernel_size=4, stride=td_downsample, padding=1)

        self.prompting_feedback = prompting_feedback

    @property
    def object_dim(self):
        return self._object_dim
    
    def downsample(self, x):
        h = w = int(x.shape[1]**0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.downsampler(x)
        return rearrange(x, 'b c h w -> b (h w) c')
    
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
            pre_feature = self.positional_embedding(feature.features, feature.positions)
        else:
            pre_feature = feature.features

        synth_prompt = None
        if self.use_synth_prompt:
            synth_prompt = self.synth_prompt(pre_feature)
        if self.td_downsample > 1:
            pre_feature = self.downsample(pre_feature)
        seed_slots, seed_attn, _ = self.slot_attention(pre_feature, conditioning, slot_mask, synth_prompt)

        q_slots, indices, commit = self.slot_pool(seed_slots)
        td_signal = td_signal_with_pos = self.synth(detach(q_slots, self.synth_in_detach), mask=seed_attn)

        if self.synth_feat_without_pos and self.positional_embedding:
            if self.prompting_feedback:
                post_feature = self.positional_embedding(feature.features + td_signal.reconstruction, feature.positions)
                td_signal_with_pos = None
            else:
                post_feature = pre_feature
                td_signal_with_pos = ocl.typing.SynthesizerOutput(
                    feedback=self.positional_embedding(td_signal.feedback, None), 
                    feedback_type=td_signal.feedback_type
                )

        slots, attn, attn_stats = self.slot_attention(post_feature, conditioning, slot_mask, td_signal_with_pos)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        if self.var_synth_in_detach:
            td_signal.reconstruction = self.synth(detach(seed_slots, True), mask=seed_attn)

        return ocl.typing.PerceptualGroupingOutput(
            slots, 
            feature_attributions=attn, 
            is_empty=slot_mask,
            pool_indices=indices,
            commit_loss=commit.sum(),
            alpha_weights=td_signal.alpha_weights,
            feedback=td_signal.feedback,
            feedback_recon=td_signal.reconstruction,
            pos_feat=feature.features,
            pre_attn=seed_attn,
            post_attn=attn,
            pre_slots=seed_slots,
            q_slots=q_slots,
            **attn_stats
        )


class SlotAttentionGroupingABSDecoding(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        pool: Union[VectorQuantize, ResidualVQ, FSQ, MemDPC],
        feat_num: int,
        feedback_type: str,
        object_decoder: Union[PatchDecoder, AutoregressivePatchDecoder],
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        synth_feat_without_pos: bool = False,
        ema_td_slot_attn: bool = True,
        ema_td_dec: bool = True,
        ema_beta: float = 0.999,
        num_blocks: int = 1,
        feedback_scale: float = 1.0,
        feedback_gating: bool = False,
        scale_sum_to_one: bool = False,
        feedback_normalization: bool = False,
        td_slot_num: Optional[int] = None,
        use_div_feedback: bool = False,
        sg_td_path: bool = True,
        prompting_feedback: bool = False,
        only_causal_past_n: Optional[int] = None,
        use_only_topk_feedback: Optional[int] = None,
        condition_sharing: bool = False,
        num_slots: int = 7,
    ):
        """Initialize Slot Attention Grouping.

        analysis-by-decoding

        no additional use of synthesizer for feedback process, it directly use decoder to generate feedback signal.

        this module should be the last module and contain object decoding

        Args:
            pool: slot pool to use: choose among VQ, RVQ, FSQ
            pool_config: slot pool configuration
            **kwargs: other arguments for SlotAttentionGrouping
        """
        super(SlotAttentionGroupingABSDecoding, self).__init__()
        self._object_dim = object_dim

        self.slot_attention = SlotAttentionWithFeedback(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
            num_blocks=num_blocks,
            feedback_scale=feedback_scale,
            feedback_gating=feedback_gating,
            scale_sum_to_one=scale_sum_to_one,
            feedback_normalization=feedback_normalization,
        )

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None
        
        self.slot_pool = pool
        self.feedback_type = feedback_type
        self.decoder = object_decoder
        self.synth_feat_without_pos = synth_feat_without_pos
        
        self.ema_td_slot_attn = ema_td_slot_attn
        self.ema_td_dec = ema_td_dec
        self.ema_beta = ema_beta

        if self.ema_td_slot_attn:
            self.pre_slot_attention = EMA(self.slot_attention, beta=self.ema_beta, update_every=10)
            self.pre_positional_embedding = EMA(self.positional_embedding, beta=self.ema_beta, update_every=10)
        else:
            self.pre_slot_attention = self.slot_attention
            self.pre_positional_embedding = self.positional_embedding
            # self.pre_positional_embedding = Sequential(
            #     DummyPositionEmbed(),
            #     build_two_layer_mlp(feature_dim, feature_dim, 4*feature_dim, initial_layer_norm=True),
            # )

        if self.ema_td_dec:
            self.synth = EMA(self.decoder, beta=self.ema_beta, update_every=10)
        else:
            self.synth = self.decoder

        self.pre_conditioning = None
        if td_slot_num is not None or condition_sharing:
            self.pre_conditioning = RandomConditioning(
                object_dim=object_dim,
                n_slots=num_slots if td_slot_num is None else td_slot_num,
            )
        
        self.use_div_feedback = use_div_feedback
        if self.use_div_feedback:
            self.div_synthesizer = DivergenceSynthesizer(object_decoder.output_dim)

        self.sg_td_path = sg_td_path

        self.prompting_feedback = prompting_feedback

        td_mask = None
        if only_causal_past_n is not None:
            num_patches = feat_num
            max_neg_value = -torch.finfo(self.slot_attention.to_q.weight.dtype).max
            td_mask = torch.triu(torch.full((num_patches, num_patches), max_neg_value), diagonal=1)
            for i in range(num_patches):
                td_mask[i, :max(i-only_causal_past_n, 0)] = max_neg_value

        self.register_buffer("td_mask", td_mask)

        self.use_only_top_k_feedback = use_only_topk_feedback

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
        """Apply slot attention based perceptual grouping.

        Args:
            feature: Features used for grouping.
            conditioning: Initial conditioning vectors for slots.
            slot_mask: Slot mask where true indicates that the slot should be masked.

        Returns:
            The grouped features.
        """
        if self.ema_td_dec:
            self.synth.update()
        if self.ema_td_slot_attn:
            self.pre_positional_embedding.update()
            self.pre_slot_attention.update()

        if self.positional_embedding:
            feature_with_pos = self.pre_positional_embedding(feature.features, feature.positions)
        else:
            feature_with_pos = feature.features

        with torch.no_grad() if self.sg_td_path else nullcontext():
            pre_conditioning = conditioning
            if self.pre_conditioning is not None:
                pre_conditioning = self.pre_conditioning(feature.features.shape[0])
            seed_slots, seed_attn, _ = self.pre_slot_attention(feature_with_pos, pre_conditioning, slot_mask)

        q_slots, indices, commit = self.slot_pool(seed_slots)

        if self.use_only_top_k_feedback is not None:
            q_2d = q_slots.view(-1, q_slots.size(-1))
            seed_2d = seed_slots.view(-1, seed_slots.size(-1))
            
            seed_q_dist = F.pairwise_distance(seed_2d, q_2d, p=2).view(q_slots.shape[:-1])
            top_k_indices = torch.topk(seed_q_dist, self.use_only_top_k_feedback, dim=-1, largest=False)[1]
            q_slots = torch.gather(q_slots, -1, repeat(top_k_indices, 'b n -> b n c', c=q_slots.size(-1)))
        
        with torch.no_grad() if self.sg_td_path else nullcontext():
            if self.td_mask is not None:
                td_signal = td_signal_with_pos = self.synth(
                    q_slots, 
                    target, 
                    input_masks=self.td_mask, 
                    image=image
                )
            else:
                td_signal = td_signal_with_pos = self.synth(
                    q_slots, 
                    target, 
                    image=image
                )
            if self.use_div_feedback:
                td_signal = self.div_synthesizer(td_signal, feature.features)
        
        if self.synth_feat_without_pos and self.positional_embedding:
            if self.prompting_feedback:
                feature_with_pos = self.positional_embedding(feature.features + td_signal.reconstruction, feature.positions)
                td_signal_with_pos = None
            else:
                feature_with_pos = self.positional_embedding(feature.features, feature.positions)
                td_signal_with_pos = ocl.typing.SynthesizerOutput(
                    feedback=self.positional_embedding(td_signal.reconstruction, None), 
                    feedback_type=self.feedback_type,
                    alpha_weights=td_signal.masks,
                )

        slots, attn, attn_stats = self.slot_attention(feature_with_pos, conditioning, slot_mask, td_signal_with_pos)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        DecodingOutput = self.decoder(slots, target, image=image)

        return ocl.typing.PerceptualGroupingDecodingOutput(
            slots, 
            object_decoder=DecodingOutput,
            feature_attributions=attn, 
            is_empty=slot_mask,
            pool_indices=indices,
            top_k_indices=top_k_indices if self.use_only_top_k_feedback is not None else None,
            commit_loss=commit.sum(),
            alpha_weights=td_signal.masks,
            feedback=td_signal.reconstruction,
            pos_feat = feature_with_pos,
            pre_attn=seed_attn,
            post_attn=attn,
            pre_slots=seed_slots,
            q_slots=q_slots,
            **attn_stats
        )

class SlotAttentionGroupingABSCADecoding(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        pool: Union[VectorQuantize, ResidualVQ, FSQ, MemDPC],
        feat_num: int,
        feedback_type: str,
        object_decoder: Union[PatchDecoder, AutoregressivePatchDecoder],
        ca_n_heads: int,
        input_feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        synth_feat_without_pos: bool = False,
        ema_td_slot_attn: bool = True,
        ema_td_dec: bool = True,
        ema_beta: float = 0.999,
        num_blocks: int = 1,
        feedback_scale: float = 1.0,
        feedback_gating: bool = False,
        scale_sum_to_one: bool = False,
        feedback_normalization: bool = False,
        td_slot_num: Optional[int] = None,
        use_div_feedback: bool = False,
        sg_td_path: bool = True,
        prompting_feedback: bool = False,
    ):
        """Initialize Slot Attention Grouping.

        analysis-by-decoding

        no additional use of synthesizer for feedback process, it directly use decoder to generate feedback signal.

        this module should be the last module and contain object decoding

        Args:
            pool: slot pool to use: choose among VQ, RVQ, FSQ
            pool_config: slot pool configuration
            **kwargs: other arguments for SlotAttentionGrouping
        """
        super(SlotAttentionGroupingABSCADecoding, self).__init__()
        self._object_dim = object_dim

        self.slot_attention = SlotAttentionWithFeedback(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
            num_blocks=num_blocks,
            feedback_scale=feedback_scale,
            feedback_gating=feedback_gating,
            scale_sum_to_one=scale_sum_to_one,
            feedback_normalization=feedback_normalization,
        )

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None
        
        self.slot_pool = pool
        self.feedback_type = feedback_type
        self.decoder = object_decoder
        self.synth_feat_without_pos = synth_feat_without_pos
        
        self.ema_td_slot_attn = ema_td_slot_attn
        self.ema_td_dec = ema_td_dec
        self.ema_beta = ema_beta

        if self.ema_td_slot_attn:
            self.pre_slot_attention = EMA(self.slot_attention, beta=self.ema_beta, update_every=10)
            self.pre_positional_embedding = EMA(self.positional_embedding, beta=self.ema_beta, update_every=10)
        else:
            self.pre_slot_attention = self.slot_attention
            self.pre_positional_embedding = self.positional_embedding
            # self.pre_positional_embedding = Sequential(
            #     DummyPositionEmbed(),
            #     build_two_layer_mlp(feature_dim, feature_dim, 4*feature_dim, initial_layer_norm=True),
            # )

        ## Cross Attention Top-Down Signal Generator
        self.synth = CrossAttn(input_feature_dim, feature_dim, ca_n_heads)

        self.pre_conditioning = None
        if td_slot_num is not None:
            self.pre_conditioning = RandomConditioning(
                object_dim=object_dim,
                n_slots=td_slot_num,
            )
        
        self.use_div_feedback = use_div_feedback
        if self.use_div_feedback:
            self.div_synthesizer = DivergenceSynthesizer(object_decoder.output_dim)

        self.sg_td_path = sg_td_path

        self.prompting_feedback = prompting_feedback

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
        """Apply slot attention based perceptual grouping.

        Args:
            feature: Features used for grouping.
            conditioning: Initial conditioning vectors for slots.
            slot_mask: Slot mask where true indicates that the slot should be masked.

        Returns:
            The grouped features.
        """
        if self.ema_td_slot_attn:
            self.pre_positional_embedding.update()
            self.pre_slot_attention.update()

        if self.positional_embedding:
            feature_with_pos = self.pre_positional_embedding(feature.features, feature.positions)
        else:
            feature_with_pos = feature.features

        with torch.no_grad() if self.sg_td_path else nullcontext():
            pre_conditioning = conditioning
            if self.pre_conditioning is not None:
                pre_conditioning = self.pre_conditioning(feature.features.shape[0])
            seed_slots, seed_attn, _ = self.pre_slot_attention(feature_with_pos, pre_conditioning, slot_mask)

        q_slots, indices, commit = self.slot_pool(seed_slots)
        
        with torch.no_grad() if self.sg_td_path else nullcontext():
            td_signal = td_signal_with_pos = self.synth(target, q_slots)
            if self.use_div_feedback:
                td_signal = self.div_synthesizer(td_signal, feature.features)
        
        if self.synth_feat_without_pos and self.positional_embedding:
            if self.prompting_feedback:
                feature_with_pos = self.positional_embedding(feature.features + td_signal.reconstruction, feature.positions)
                td_signal_with_pos = None
            else:
                feature_with_pos = self.positional_embedding(feature.features, feature.positions)
                td_signal_with_pos = ocl.typing.SynthesizerOutput(
                    feedback=self.positional_embedding(td_signal.reconstruction, None), 
                    feedback_type=self.feedback_type,
                    alpha_weights=td_signal.masks,
                )

        slots, attn, attn_stats = self.slot_attention(feature_with_pos, conditioning, slot_mask, td_signal_with_pos)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        DecodingOutput = self.decoder(slots, target, image=image)

        return ocl.typing.PerceptualGroupingDecodingOutput(
            slots, 
            object_decoder=DecodingOutput,
            feature_attributions=attn, 
            is_empty=slot_mask,
            pool_indices=indices,
            commit_loss=commit.sum(),
            alpha_weights=td_signal.masks,
            feedback=td_signal.reconstruction,
            pos_feat = feature_with_pos,
            pre_attn=seed_attn,
            post_attn=attn,
            pre_slots=seed_slots,
            q_slots=q_slots,
            **attn_stats
        )

class StickBreakingGrouping(nn.Module):
    """Perceptual grouping based on a stick-breaking process.

    The idea is to pick a random feature from a yet unexplained part of the feature map, then see
    which parts of the feature map are "explained" by this feature using a kernel distance. This
    process is iterated until some termination criterion is reached. In principle, this process
    allows to extract a variable number of slots per image.

    This is based on Engelcke et al, GENESIS-V2: Inferring Unordered Object Representations without
    Iterative Refinement, http://arxiv.org/abs/2104.09958. Our implementation here differs a bit from
    the one described there:

    - It only implements one kernel distance, the Gaussian kernel
    - It does not take features positions into account when computing the kernel distances
    - It L2-normalises the input features to get comparable scales of the kernel distance
    - It has multiple termination criteria, namely termination based on fraction explained, mean
      mask value, and min-max mask value. GENESIS-V2 implements termination based on mean mask
      value, but does not mention it in the paper. Note that by default, all termination criteria
      are disabled.
    """

    def __init__(
        self,
        object_dim: int,
        feature_dim: int,
        n_slots: int,
        kernel_var: float = 1.0,
        learn_kernel_var: bool = False,
        max_unexplained: float = 0.0,
        min_slot_mask: float = 0.0,
        min_max_mask_value: float = 0.0,
        early_termination: bool = False,
        add_unexplained: bool = False,
        eps: float = 1e-8,
        detach_features: bool = False,
        use_input_layernorm: bool = False,
    ):
        """Initialize stick-breaking-based perceptual grouping.

        Args:
            object_dim: Dimensionality of extracted slots.
            feature_dim: Dimensionality of features to operate on.
            n_slots: Maximum number of slots.
            kernel_var: Variance in Gaussian kernel.
            learn_kernel_var: Whether kernel variance should be included as trainable parameter.
            max_unexplained: If fraction of unexplained features drops under this value,
                drop the slot.
            min_slot_mask: If slot mask has lower average value than this value, drop the slot.
            min_max_mask_value: If slot mask's maximum value is lower than this value,
                drop the slot.
            early_termination: If true, all slots after the first dropped slot are also dropped.
            add_unexplained: If true, add a slot that covers all unexplained parts at the point
                when the first slot was dropped.
            eps: Minimum value for masks.
            detach_features: If true, detach input features such that no gradient flows through
                this operation.
            use_input_layernorm: Apply layernorm to features prior to grouping.
        """
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = object_dim

        assert kernel_var > 0.0
        if learn_kernel_var:
            self.kernel_logvar = nn.Parameter(torch.tensor(math.log(kernel_var)))
        else:
            self.register_buffer("kernel_logvar", torch.tensor(math.log(kernel_var)))

        assert 0.0 <= max_unexplained < 1.0
        self.max_unexplained = max_unexplained
        assert 0.0 <= min_slot_mask < 1.0
        self.min_slot_mask = min_slot_mask
        assert 0.0 <= min_max_mask_value < 1.0
        self.min_max_mask_value = min_max_mask_value

        self.early_termination = early_termination
        self.add_unexplained = add_unexplained
        if add_unexplained and not early_termination:
            raise ValueError("`add_unexplained=True` only works with `early_termination=True`")

        self.eps = eps
        self.log_eps = math.log(eps)
        self.detach_features = detach_features

        if use_input_layernorm:
            self.in_proj = nn.Sequential(
                nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim)
            )
            torch.nn.init.xavier_uniform_(self.in_proj[-1].weight)
            torch.nn.init.zeros_(self.in_proj[-1].bias)
        else:
            self.in_proj = nn.Linear(feature_dim, feature_dim)
            torch.nn.init.xavier_uniform_(self.in_proj.weight)
            torch.nn.init.zeros_(self.in_proj.bias)

        self.out_proj = nn.Linear(feature_dim, object_dim)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        torch.nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, features: ocl.typing.FeatureExtractorOutput
    ) -> ocl.typing.PerceptualGroupingOutput:
        """Apply stick-breaking-based perceptual grouping to input features.

        Args:
            features: Features that should be grouped.

        Returns:
            Grouped features.
        """
        features = features.features
        bs, n_features, feature_dim = features.shape
        if self.detach_features:
            features = features.detach()

        proj_features = torch.nn.functional.normalize(self.in_proj(features), dim=-1)

        # The scope keep tracks of the unexplained parts of the feature map
        log_scope = torch.zeros_like(features[:, :, 0])
        # Seeds are used for random sampling of features
        log_seeds = torch.rand_like(log_scope).clamp_min(self.eps).log()

        slot_masks = []
        log_scopes = []

        # Always iterate for `n_iters` steps for batching reasons. Termination is modeled afterwards.
        n_iters = self.n_slots - 1 if self.add_unexplained else self.n_slots
        for _ in range(n_iters):
            log_scopes.append(log_scope)

            # Sample random features from unexplained parts of the feature map
            rand_idxs = torch.argmax(log_scope + log_seeds, dim=1)
            cur_centers = proj_features.gather(
                1, rand_idxs.view(bs, 1, 1).expand(-1, -1, feature_dim)
            )

            # Compute similarity between selected features and other features. alpha can be
            # considered an attention mask.
            dists = torch.sum((cur_centers - proj_features) ** 2, dim=-1)
            log_alpha = (-dists / self.kernel_logvar.exp()).clamp_min(self.log_eps)

            # To get the slot mask, we subtract already explained parts from alpha using the scope
            mask = (log_scope + log_alpha).exp()
            slot_masks.append(mask)

            # Update scope by masking out parts explained by the current iteration
            log_1m_alpha = (1 - log_alpha.exp()).clamp_min(self.eps).log()
            log_scope = log_scope + log_1m_alpha

        if self.add_unexplained:
            slot_masks.append(log_scope.exp())
            log_scopes.append(log_scope)

        slot_masks = torch.stack(slot_masks, dim=1)
        scopes = torch.stack(log_scopes, dim=1).exp()

        # Compute criteria for ignoring slots
        empty_slots = torch.zeros_like(slot_masks[:, :, 0], dtype=torch.bool)
        # When fraction of unexplained features drops under threshold, ignore slot,
        empty_slots |= scopes.mean(dim=-1) < self.max_unexplained
        # or when slot's mean mask is under threshold, ignore slot,
        empty_slots |= slot_masks.mean(dim=-1) < self.min_slot_mask
        # or when slot's masks maximum value is under threshold, ignore slot.
        empty_slots |= slot_masks.max(dim=-1).values < self.min_max_mask_value

        if self.early_termination:
            # Simulate early termination by marking all slots after the first empty slot as empty
            empty_slots = torch.cummax(empty_slots, dim=1).values
            if self.add_unexplained:
                # After termination, add one more slot using the unexplained parts at that point
                first_empty = torch.argmax(empty_slots.to(torch.int32), dim=1).unsqueeze(-1)
                empty_slots.scatter_(1, first_empty, torch.zeros_like(first_empty, dtype=torch.bool))

                idxs = first_empty.view(bs, 1, 1).expand(-1, -1, n_features)
                unexplained = scopes.gather(1, idxs)
                slot_masks.scatter_(1, idxs, unexplained)

        # Create slot representations as weighted average of feature map
        slots = torch.einsum("bkp,bpd->bkd", slot_masks, features)
        slots = slots / slot_masks.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        slots = self.out_proj(slots)

        # Zero-out masked slots
        slots.masked_fill_(empty_slots.view(bs, slots.shape[1], 1), 0.0)

        return ocl.typing.PerceptualGroupingOutput(
            slots, feature_attributions=slot_masks, is_empty=empty_slots
        )


class KMeansGrouping(nn.Module):
    """Simple K-means clustering based grouping."""

    def __init__(
        self,
        n_slots: int,
        use_l2_normalization: bool = True,
        clustering_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._object_dim = None
        self.n_slots = n_slots
        self.use_l2_normalization = use_l2_normalization

        kwargs = clustering_kwargs if clustering_kwargs is not None else {}
        self.make_clustering = lambda: cluster.KMeans(n_clusters=n_slots, **kwargs)

    @property
    def object_dim(self):
        return self._object_dim

    def forward(
        self, feature: ocl.typing.FeatureExtractorOutput
    ) -> ocl.typing.PerceptualGroupingOutput:
        feature = feature.features
        if self._object_dim is None:
            self._object_dim = feature.shape[-1]

        if self.use_l2_normalization:
            feature = torch.nn.functional.normalize(feature, dim=-1)

        batch_features = feature.detach().cpu().numpy()

        cluster_ids = []
        cluster_centers = []

        for feat in batch_features:
            clustering = self.make_clustering()

            cluster_ids.append(clustering.fit_predict(feat).astype(numpy.int64))
            cluster_centers.append(clustering.cluster_centers_)

        cluster_ids = torch.from_numpy(numpy.stack(cluster_ids))
        cluster_centers = torch.from_numpy(numpy.stack(cluster_centers))

        slot_masks = torch.nn.functional.one_hot(cluster_ids, num_classes=self.n_slots)
        slot_masks = slot_masks.transpose(-2, -1).to(torch.float32)

        return ocl.typing.PerceptualGroupingOutput(
            cluster_centers.to(feature.device), feature_attributions=slot_masks.to(feature.device)
        )


class SlotAttentionGroupingABSDecodingIterQuant(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        pool: Union[VectorQuantize, ResidualVQ, FSQ, MemDPC],
        feat_num: int,
        feedback_type: str,
        object_decoder: Union[PatchDecoder, AutoregressivePatchDecoder],
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        synth_feat_without_pos: bool = False,
        ema_td_model: bool = True,
        ema_beta: float = 0.999,
        num_blocks: int = 1,
    ):
        """Initialize Slot Attention Grouping.

        analysis-by-decoding

        no additional use of synthesizer for feedback process, it directly use decoder to generate feedback signal.

        this module should be the last module and contain object decoding

        Args:
            pool: slot pool to use: choose among VQ, RVQ, FSQ
            pool_config: slot pool configuration
            **kwargs: other arguments for SlotAttentionGrouping
        """
        super().__init__()
        self._object_dim = object_dim

        self.slot_attention = SlotAttentionWithFeedbackIterQuant(
            dim=object_dim,
            feature_dim=feature_dim,
            pool=pool,
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
        
        self.feedback_type = feedback_type
        self.decoder = object_decoder
        self.synth_feat_without_pos = synth_feat_without_pos
        
        self.ema_td_model = ema_td_model
        self.ema_beta = ema_beta
        if ema_td_model:
            self.pre_slot_attention = EMA(self.slot_attention, beta=self.ema_beta, update_every=10, ignore_startswith_names=set('pool'))
            self.pre_positional_embedding = EMA(self.positional_embedding, beta=self.ema_beta, update_every=10)
            self.synth = EMA(self.decoder, beta=self.ema_beta, update_every=10)
        else:
            self.pre_slot_attention = self.slot_attention
            self.pre_positional_embedding = self.positional_embedding
            self.synth = self.decoder

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
        """Apply slot attention based perceptual grouping.

        Args:
            feature: Features used for grouping.
            conditioning: Initial conditioning vectors for slots.
            slot_mask: Slot mask where true indicates that the slot should be masked.

        Returns:
            The grouped features.
        """
        if self.ema_td_model:
            self.pre_slot_attention.update()
            self.pre_positional_embedding.update()
            self.synth.update()

        if self.positional_embedding:
            feature_with_pos = self.pre_positional_embedding(feature.features, feature.positions)
        else:
            feature_with_pos = feature.features

        seed_slots, seed_attn, _, pool_stats = self.pre_slot_attention(feature_with_pos, conditioning, slot_mask)
        indices, commit = pool_stats['indices'], pool_stats['commit']

        with torch.no_grad():
            td_signal = td_signal_with_pos = self.synth(seed_slots)
        
        if self.synth_feat_without_pos and self.positional_embedding:
            td_signal_with_pos = ocl.typing.SynthesizerOutput(
                feedback=self.positional_embedding(td_signal.reconstruction, None), 
                feedback_type=self.feedback_type
            )

        if self.positional_embedding and self.ema_td_model:
            feature_with_pos = self.positional_embedding(feature.features, feature.positions)
        slots, attn, attn_stats, _ = self.slot_attention(feature_with_pos, conditioning, slot_mask, td_signal_with_pos)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        DecodingOutput = self.decoder(slots, target, image)

        return ocl.typing.PerceptualGroupingDecodingOutput(
            slots, 
            object_decoder=DecodingOutput,
            feature_attributions=attn, 
            is_empty=slot_mask,
            pool_indices=indices,
            commit_loss=commit.sum(),
            alpha_weights=td_signal_with_pos.alpha_weights,
            feedback=td_signal.reconstruction,
            pos_feat = feature_with_pos,
            **attn_stats
        )
    

class SlotAttentionGroupingABSIterQuantSynth(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        pool: Union[VectorQuantize, ResidualVQ, FSQ],
        feat_num: int,
        feedback_type: str,
        synthesizer: Union[BraodcastingSynthesizer, MaskedBroadcastingSynthesizer],
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        synth_feat_without_pos: bool = False,
        use_synth_prompt: bool = False,
        synth_in_detach: bool = False,
        num_blocks: int = 1,
    ):
        """Initialize Slot Attention Grouping.

        Args:
            pool: slot pool to use: choose among VQ, RVQ, FSQ
            pool_config: slot pool configuration
            **kwargs: other arguments for SlotAttentionGrouping
        """
        super().__init__()
        self._object_dim = object_dim
        self.slot_attention = SlotAttentionWithFeedbackIterQuantSynth(
            dim=object_dim,
            feature_dim=feature_dim,
            pool=pool,
            synth=synthesizer,
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

        slots, attn, attn_stats, pool_stats, td_signal = self.slot_attention(feature, conditioning, slot_mask)

        indices, commit = pool_stats['indices'], pool_stats['commit']

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingOutput(
            slots, 
            feature_attributions=attn, 
            is_empty=slot_mask,
            pool_indices=indices,
            commit_loss=commit.sum(),
            alpha_weights=td_signal.alpha_weights,
            feedback=td_signal.feedback,
            feedback_recon=td_signal.reconstruction,
            pos_feat = feature,
            **attn_stats
        )
    

class SlotAttentionGroupingHyper(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        pool: Union[VectorQuantize, ResidualVQ, FSQ],
        feat_num: int,
        hypernet: Hypernet,
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
        td_downsample: int = 1,
    ):
        """Initialize Slot Attention Grouping.

        Args:
            pool: slot pool to use: choose among VQ, RVQ, FSQ
            pool_config: slot pool configuration
            **kwargs: other arguments for SlotAttentionGrouping
        """
        super().__init__()
        self._object_dim = object_dim
        self.slot_attention = SlotAttentionWithFeedbackHyper(
            dim=object_dim,
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
        
        self.slot_pool = pool
        self.hypernet = hypernet

        self.td_downsample = td_downsample
        self.downsampler = None
        if self.td_downsample > 1:
            self.downsampler = nn.Conv2d(feature_dim, feature_dim, kernel_size=4, stride=td_downsample, padding=1)

    @property
    def object_dim(self):
        return self._object_dim
    
    def downsample(self, x):
        h = w = int(x.shape[1]**0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.downsampler(x)
        return rearrange(x, 'b c h w -> b (h w) c')
    
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

        td_feature = feature
        if self.td_downsample > 1:
            td_feature = self.downsample(feature)
        seed_slots, seed_attn, _ = self.slot_attention(td_feature, conditioning, slot_mask)

        q_slots, indices, commit = self.slot_pool(seed_slots)
        td_signal = self.hypernet(q_slots)

        slots, attn, attn_stats = self.slot_attention(feature, conditioning, slot_mask, td_signal)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingOutput(
            slots, 
            feature_attributions=attn, 
            is_empty=slot_mask,
            pool_indices=indices,
            commit_loss=commit.sum(),
            pos_feat = feature,
            pre_attn=seed_attn,
            post_attn=attn,
            **attn_stats
        )


class SlotAttentionGroupingVQ(nn.Module):
    """Implementation of SlotAttention for perceptual grouping."""

    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        pool: Union[VectorQuantize, ResidualVQ, FSQ, MemDPC],
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
    ):
        """Initialize Slot Attention Grouping.

        Args:
            feature_dim: Dimensionality of features to slot attention (after positional encoding).
            object_dim: Dimensionality of slots.
            kvq_dim: Dimensionality after projecting to keys, values, and queries. If `None`,
                `object_dim` is used.
            n_heads: Number of heads slot attention uses.
            iters: Number of slot attention iterations.
            eps: Epsilon in slot attention.
            ff_mlp: Optional module applied slot-wise after GRU update.
            positional_embedding: Optional module applied to the features before slot attention,
                adding positional encoding.
            use_projection_bias: Whether to use biases in key, value, query projections.
            use_implicit_differentiation: Whether to use implicit differentiation trick. If true,
                performs one more iteration of slot attention that is used for the gradient step
                after `iters` iterations of slot attention without gradients. Faster and more memory
                efficient than the standard version, but can not backpropagate gradients to the
                conditioning input.
            use_empty_slot_for_masked_slots: Replace slots masked with a learnt empty slot vector.
        """
        super().__init__()
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

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

        self.pool = pool

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

        slots, pre_attn = self.slot_attention(feature, conditioning, slot_mask)

        q_slots, indices, commit = self.pool(slots)

        slots, post_attn = self.slot_attention(feature, q_slots, slot_mask)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingOutput(
            slots, feature_attributions=post_attn, pre_attn=pre_attn, is_empty=slot_mask, pool_indices=indices,
            commit_loss=commit.sum(), pre_slots=slots
        )
