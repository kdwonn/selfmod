import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, reduce, repeat, rearrange
from typing import Union, Callable, Optional, Tuple
import math


class ZeroModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return torch.zeros(x.shape, device=x.device, dtype=x.dtype)


class AdaLoRA(nn.Module):
    def __init__(self, dim, rank, num_entries, scale, reduction=False, out_dim=None):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.out_dim = dim if out_dim is None else out_dim
        self.down_proj_values = nn.Parameter(torch.zeros(num_entries, dim, rank))
        self.up_proj_values = nn.Parameter(torch.zeros(num_entries, rank, self.out_dim))
        self.reduction = reduction
        self.scale = scale / math.sqrt(self.rank)

        # Same with microsoft official implementation
        nn.init.kaiming_uniform_(self.down_proj_values, a=math.sqrt(5))
        # nn.init.normal_(self.down_proj_values, std=1/self.rank) # following paper
        nn.init.zeros_(self.up_proj_values)
    
    def forward(self, slots, indices):
        down_projs = self.down_proj_values[indices, :, :]
        up_projs = self.up_proj_values[indices, :, :]

        if self.reduction:
            down_projs = repeat(
                down_projs.mean(dim=1), 'b d r -> b k d r', k=slots.shape[1])
            up_projs = repeat(
                up_projs.mean(dim=1), 'b d r -> b k d r', k=slots.shape[1])

        down_slots = einsum(slots, down_projs, 'b k d, b k d r -> b k r')
        out_slots = einsum(down_slots, up_projs, 'b k r, b k r d -> b k d')

        return out_slots * self.scale


class AdaLN(nn.Module):
    def __init__(self, dim, num_entries, reduction=False):
        super().__init__()
        self.dim = dim
        self.num_entries = num_entries
        self.reduction = reduction
        self.scale_values = nn.Parameter(torch.zeros(num_entries, dim))
        self.bias_values = nn.Parameter(torch.zeros(num_entries, dim))

        nn.init.zeros_(self.scale_values)
        nn.init.zeros_(self.bias_values)

    def forward(self, slots, indices):
        scale = self.scale_values[indices, :]
        bias = self.bias_values[indices, :]

        if self.reduction:
            scale = repeat(
                scale.mean(dim=1), 'b d -> b k d', k=scale.shape[1])
            bias = repeat(
                bias.mean(dim=1), 'b d -> b k d', k=bias.shape[1])

        out_slots = F.layer_norm(slots, slots.shape, weight=scale, bias=bias)

        return out_slots
    

class AdaScaling(nn.Module):
    def __init__(self, dim, num_entries, reduction=False) -> None:
        super().__init__()
        self.dim = dim
        self.num_entries = num_entries
        self.reduction = reduction
        self.scale_values = nn.Parameter(torch.zeros(num_entries, dim))
        nn.init.zeros_(self.scale_values)

    def forward(self, slots, indices):
        scale = self.scale_values[indices, :]

        if self.reduction:
            scale = repeat(
                scale.mean(dim=1), 'b d -> b k d', k=scale.shape[1])
        if len(slots.shape) == 4:
            #  when using multi-head
            scale = rearrange(scale, 'b k d -> b k () d')
        out_slots = scale * slots

        return out_slots
    

class AdaMLP(nn.Module):
    def __init__(self, dim, mul, num_entries, reduction=False):
        super().__init__()
        self.dim = dim
        self.mul = mul
        self.w1 = nn.Parameter(torch.zeros(num_entries, dim, mul*dim))
        self.b1 = nn.Parameter(torch.zeros(num_entries, mul*dim))
        self.w2 = nn.Parameter(torch.zeros(num_entries, mul*dim, dim))
        self.b2 = nn.Parameter(torch.zeros(num_entries, dim))
        self.reduction = reduction

        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)
    
    def forward(self, slots, indices):
        w1 = self.w1[indices, :, :]
        b1 = self.b1[indices, :]
        w2 = self.w2[indices, :, :]
        b2 = self.b2[indices, :]

        if self.reduction:
            w1 = repeat(
                w1.mean(dim=1), 'b d r -> b k d r', k=slots.shape[1])
            w2 = repeat(
                w2.mean(dim=1), 'b d r -> b k d r', k=slots.shape[1])

        slots = einsum(slots, w1, 'b k d, b k d r -> b k r')
        slots = slots + b1
        slots = F.relu(slots)
        slots = einsum(slots, w2, 'b k r, b k r d -> b k d')
        slots = slots + b2

        return slots
    

class PromptTo2D(nn.Module):
    def __init__(self, dim, num_entries, use_soft=False) -> None:
        super().__init__()
        self.dim = dim
        self.num_entries = num_entries

        self.prompt = nn.Parameter(torch.zeros(num_entries, dim))
        nn.init.xavier_normal_(self.prompt)

        self.soft = use_soft

    def forward(self, indices, attn_map):
        prompts = self.prompt[indices]
        if self.soft:
            masked_broadcast = einsum(attn_map, prompts, 'b k n, b k d -> b n d')
        else:
            argmax_mask = torch.argmax(attn_map, dim=1)
            masked_broadcast = torch.gather(prompts, 1, argmax_mask.unsqueeze(-1).repeat(1, 1, prompts.shape[-1])) # B x HW -> B x HW x D
        return masked_broadcast
    

class GumbelAdaLoRA(AdaLoRA):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def log(self, t, eps = 1e-20):
        return torch.log(t.clamp(min = eps))

    def gumbel_noise(self,t):
        noise = torch.zeros_like(t).uniform_(0, 1)
        return -self.log(-self.log(noise))

    def gumbel_sample(
        self,
        logits,
        temperature = 1.,
        stochastic = True,
        reinmax = False,
        dim = -1
    ):
        dtype, size = logits.dtype, logits.shape[dim]

        if self.training and stochastic and temperature > 0:
            sampling_logits = (logits / temperature) + self.gumbel_noise(logits)
        else:
            sampling_logits = logits

        ind = sampling_logits.argmax(dim = dim)
        one_hot = F.one_hot(ind, size).type(dtype)

        # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
        if reinmax:
            pi0 = logits.softmax(dim = dim)
            pi1 = (one_hot + (logits / temperature).softmax(dim = dim)) / 2
            pi1 = ((self.log(pi1) - logits).detach() + logits).softmax(dim = 1)
            pi2 = 2 * pi1 - 0.5 * pi0
            one_hot = pi2 - pi2.detach() + one_hot
        else:
            pi1 = (logits / temperature).softmax(dim = dim)
            one_hot = one_hot + pi1 - pi1.detach()

        return ind, one_hot

    def forward(self, x, distances, temperature):
        assert len(x.shape)==3
        if len(distances.shape)==4:
            distances = rearrange(distances, 'h b k n -> (h b) k n')
        ind, selection = self.gumbel_sample(distances, temperature=temperature)

        down_x = einsum(x, self.down_proj_values, 'b k d, n d r -> b k n r')
        up_x = einsum(down_x, self.up_proj_values, 'b k n r, n r d -> b k n d')

        if self.reduction:
            selection = repeat(selection.mean(dim=1), 'b n -> b k n', k=x.shape[1])

        selected = einsum(selection, up_x, 'b k n, b k n d -> b k d')

        return selected * self.scale