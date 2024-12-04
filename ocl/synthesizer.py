import torch.nn as nn
import torch.nn.functional as F
import torch
from math import sqrt
from einops import rearrange, repeat, einsum, reduce
from ocl.typing import SynthesizerOutput
from typing import Union
from ocl.neural_networks import build_two_layer_mlp
from ocl.decoding import PatchReconstructionOutput


class BraodcastingSynthesizer(nn.Module):
    def __init__(self, slot_dim, feat_dim, feat_shape, layer_num=2, feedback_type='v', slotwise=False) -> None:
        super().__init__()
        self.layers = []
        for i in range(layer_num-1):
            self.layers.append(nn.Linear(slot_dim, slot_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(slot_dim, feat_dim+1))
        self.mlp = nn.Sequential(*self.layers)
        self.normalizer = F.softmax
        self.feat_shape, self.h  = feat_shape, int(sqrt(feat_shape))
        # self.pos_emb = PositionalEmbedding(self.h, self.h, slot_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, feat_shape, slot_dim) * 0.02)
        self.feedback_type = feedback_type
        self.slotwise = slotwise

    def forward(self, slots, mask, target=None):
        _, s, _ = slots.shape
        axes = {'h': self.h, 'w': self.h, 's': s}
        slots = repeat(slots, 'b s d -> b s (h w) d', **axes)
        slots = slots + self.pos_embed
        feat_decode = self.mlp(slots)
        feat, alpha = feat_decode[:, :, :, :-1], feat_decode[:, :, :, -1]
        alpha = self.normalizer(alpha, dim=1)
        recon = einsum(feat, alpha, 'b s hw d, b s hw -> b hw d')
        if self.slotwise:
            feedback = einsum(feat, alpha, 'b s hw d, b s hw -> b s hw d')
        else:
            feedback = recon
        # return recon, rearrange(alpha, 'b s (h w) -> b s () h w', **axes)
        return SynthesizerOutput(
            feedback=feedback,
            reconstruction=recon, 
            alpha_weights=alpha, 
            feedback_type=self.feedback_type
        )
    

class MaskedBroadcastingSynthesizer(nn.Module):
    def __init__(self, slot_dim, feat_dim, feat_shape, layer_num=2, feedback_type='v') -> None:
        super().__init__()
        self.layers = []
        for i in range(layer_num-1):
            self.layers.append(nn.Linear(slot_dim, slot_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(slot_dim, feat_dim))
        self.mlp = nn.Sequential(*self.layers)
        self.pos_embed = nn.Parameter(torch.randn(1, feat_shape, slot_dim) * 0.02)
        self.feedback_type = feedback_type

    def forward(self, slots, mask, target=None):
        argmax_mask = torch.argmax(mask, dim=1) # B x O x HW -> B x HW
        masked_broadcast = torch.gather(slots, 1, argmax_mask.unsqueeze(-1).repeat(1, 1, slots.shape[-1])) # B x HW -> B x HW x D
        masked_broadcast = masked_broadcast + self.pos_embed
        recon = self.mlp(masked_broadcast)
        return SynthesizerOutput(
            feedback=recon,
            reconstruction=recon, 
            alpha_weights=mask, 
            feedback_type=self.feedback_type
        )
    

class SynthPrompt(nn.Module):
    def __init__(self, feat_dim, feat_shape, feedback_type="v") -> None:
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(feat_shape, feat_dim))
        nn.init.xavier_uniform_(self.prompt)
        self.feedback_type = feedback_type

    def forward(self, x):
        repeat(self.prompt, 'n d -> b n d', b=x.shape[0])
        return SynthesizerOutput(
            feedback=self.prompt,
            reconstruction=self.prompt, 
            feedback_type=self.feedback_type
        )
    

class SoftMaskedBroadcastingSynthesizer(MaskedBroadcastingSynthesizer):
    def __init__(self, slot_dim, feat_dim, feat_shape, layer_num=2, feedback_type='v') -> None:
        super().__init__(slot_dim, feat_dim, feat_shape, layer_num, feedback_type)

    def forward(self, slots, mask, target=None):
        masked_broadcast = einsum(slots, mask, "b s d, b s hw -> b hw d")
        masked_broadcast = masked_broadcast + self.pos_embed
        recon = self.mlp(masked_broadcast)
        return SynthesizerOutput(
            feedback=recon,
            reconstruction=recon, 
            alpha_weights=mask, 
            feedback_type=self.feedback_type
        )


class DivergenceSynthesizer(nn.Module):
    def __init__(self, feat_dim) -> None:
        super().__init__()
        self.mlp = build_two_layer_mlp(feat_dim, feat_dim, feat_dim, initial_layer_norm=True)

    def forward(self, synth_output: PatchReconstructionOutput, target: torch.Tensor):
        div = synth_output.reconstruction - target
        div = self.mlp(div)
        return PatchReconstructionOutput(div, masks=synth_output.masks)
    

class MaskedFeatSumSynthesizer(nn.Module):
    def __init__(self, detach_mask=True) -> None:
        super().__init__()
        self.detach_mask = detach_mask
        self.eps = 1e-8

    def forward(self, slots, mask, features):
        if self.detach_mask:
            mask = mask.detach()
        masked_feat = einsum(features, mask, "b n d, b k n -> b k d")
        masked_feat = masked_feat / (mask + self.eps).sum(dim=-1, keepdim=True)
        return masked_feat
    

class CodeToFeatAttn(nn.Module):
    def __init__(self, dim=1, one_hot=False, temp=1) -> None:
        super().__init__()
        self.dim = dim
        self.one_hot = one_hot
        self.temp = temp

    def forward(self, code, features):
        attn = einsum(code, features, "b k d, b n d -> b k n")
        attn = (attn / self.temp).softmax(dim=self.dim)

        if self.one_hot:
            attn = attn.argmax(dim=self.dim)
            attn = F.one_hot(attn, num_classes=code.shape[1])
            attn = rearrange(attn, "b n k -> b k n").to(code.dtype)

        return attn