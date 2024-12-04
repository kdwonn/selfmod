import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, reduce, repeat, rearrange
import torch.nn as nn
import math


class MaskedAdaGN(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_groups = 24,
        eps = 1e-6,
        pre_ln = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps

        self.to_scale = nn.Linear(in_dim, out_dim, bias=False)
        self.to_bias = nn.Linear(in_dim, out_dim, bias=False)

        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_dim, eps=eps, affine=False)

        nn.init.kaiming_uniform_(self.to_scale.weight, a=math.sqrt(5))
        nn.init.zeros_(self.to_bias.weight)

        self.ln_first = pre_ln
        if self.ln_first:
            self.scale_ln = nn.LayerNorm(in_dim, eps=eps)
            self.bias_ln = nn.LayerNorm(in_dim, eps=eps)
        else:
            self.scale_ln = nn.LayerNorm(out_dim, eps=eps)
            self.bias_ln = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, inputs, codes, masks):
        if self.ln_first:
            scale = self.to_scale(self.scale_ln(codes))
            bias = self.to_bias(self.scale_ln(codes))
        else:
            scale = self.scale_ln(self.to_scale(codes))
            bias = self.bias_ln(self.to_bias(codes))

        masks = torch.argmax(masks, dim=1) # B x HW
        masked_scale = torch.gather(scale, 1, masks.unsqueeze(-1).repeat(1, 1, scale.shape[-1]))
        masked_bias = torch.gather(bias, 1, masks.unsqueeze(-1).repeat(1, 1, bias.shape[-1]))

        out = self.group_norm(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        out = out * masked_scale + masked_bias

        return out
    

class MaskedChannelWeighting(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ln = nn.LayerNorm(out_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, inputs, codes, masks):
        weights = self.ln(self.proj(codes))
        
        masks = torch.argmax(masks, dim=1) # B x HW
        masked_weights = torch.gather(weights, 1, masks.unsqueeze(-1).repeat(1, 1, weights.shape[-1]))

        return inputs * masked_weights


class MaskedAddition(nn.Module):
    def __init__(self, in_dim, out_dim, use_mlp=False):
        super().__init__()
        self.ln = nn.LayerNorm(out_dim)
        if use_mlp:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
        else:
            self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, inputs, codes, masks):
        weights = self.ln(self.proj(codes))
        
        masks = torch.argmax(masks, dim=1) # B x HW
        masked_weights = torch.gather(weights, 1, masks.unsqueeze(-1).repeat(1, 1, weights.shape[-1]))

        return masked_weights


class MaskedDispatch(nn.Module):
    def __init__(self, in_dim, out_dim, last_ln=False, soft_dispatch=False, scale_after_fc=False, detach_mask=True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if last_ln:
            self.fc = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.soft_dispatch = soft_dispatch
        self.scale_after_fc = scale_after_fc
        self.detach_mask = detach_mask

    def forward(self, inputs, codes, masks):
        # mask shape: b x k x hw
        # codes shape: b x k x d
        # inputs shape: b x hw x d
        if self.detach_mask:
            masks = masks.detach()
        if self.soft_dispatch:
            sim_to_code = F.sigmoid(self.scale * masks + self.bias)
            codes = repeat(codes, 'b k d -> b k hw d', hw=inputs.shape[1])
            ret = self.fc(sim_to_code.unsqueeze(-1) * codes).sum(dim=1)
        else:
            assignment = torch.argmax(masks, dim=1) # B x HW
            codes = torch.gather(codes, 1, assignment.unsqueeze(-1).repeat(1, 1, codes.shape[-1]))
            sim_to_code = torch.gather(
                masks, 1, assignment.unsqueeze(1)
            ).squeeze(1)
            sim_to_code = F.sigmoid(self.scale * sim_to_code + self.bias)
            if self.scale_after_fc:
                ret = einsum(sim_to_code, self.fc(codes), 'b n, b n d -> b n d')
            else:
                ret = self.fc(einsum(sim_to_code, codes, 'b n, b n d -> b n d'))

        return ret
    

class DiscreteMaskedAdaGN(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_entries,
        num_groups = 24,
        eps = 1e-6,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps

        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_dim, eps=eps, affine=False)

        self.scales = nn.Parameter(torch.ones(num_entries, out_dim))
        self.bias = nn.Parameter(torch.zeros(num_entries, out_dim))

    def forward(self, inputs, indices=None, masks=None):
        if masks is None and indices is None:
            scale = reduce(self.scales, 'n d -> d', 'mean')
            bias = reduce(self.bias, 'n d -> d', 'mean')
            out = self.group_norm(inputs.permute(0, 2, 1)).permute(0, 2, 1)
            out = out * scale + bias
        else: 
            masks = torch.argmax(masks, dim=1) # B x HW

            scales = self.scales[indices]
            biases = self.bias[indices]

            masked_scale = torch.gather(scales, 1, masks.unsqueeze(-1).repeat(1, 1, scales.shape[-1]))
            masked_bias = torch.gather(biases, 1, masks.unsqueeze(-1).repeat(1, 1, biases.shape[-1]))

            out = self.group_norm(inputs.permute(0, 2, 1)).permute(0, 2, 1)
            out = out * masked_scale + masked_bias

        return out

class GroupNorm(nn.Module):
    def __init__(
        self,
        dim,
        num_groups = 24,
        eps = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        self.eps = eps

        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim, eps=eps, affine=True)

    def forward(self, inputs):
        return self.group_norm(inputs.permute(0, 2, 1)).permute(0, 2, 1)