import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses

@dataclasses.dataclass
class HypernetOutput:
    ln_w: torch.Tensor
    ln_b: torch.Tensor
    ada_down_proj: torch.Tensor
    ada_up_proj: torch.Tensor
    ada_ln_w: torch.Tensor
    ada_ln_b: torch.Tensor
    use_ada: bool

class Hypernet(nn.Module):
    def __init__(self, feat_dim, slot_dim, down_proj_dim, use_ada) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.slot_dim = slot_dim
        self.down_proj_dim = down_proj_dim

        self.set_rep = nn.Linear(slot_dim, slot_dim)

        self.feat_ln_hypernet = nn.Linear(slot_dim, 2 * feat_dim)
        self.ada_ln_hypernet = nn.Linear(slot_dim, 2 * slot_dim)
        self.ada_down_hypernet = nn.Linear(slot_dim, slot_dim * down_proj_dim)
        self.ada_up_hypernet = nn.Linear(slot_dim, down_proj_dim * slot_dim)

        self.use_ada = use_ada

    def encode_slot_set_rep(self, slots):
        slots = self.set_rep(slots)
        return F.gelu(slots.sum(dim=1).unsqueeze(1))

    def forward(self, codes):
        ln_w, ln_bias = self.feat_ln_hypernet(codes).chunk(2, dim=-1)

        slot_set_rep = self.encode_slot_set_rep(codes)

        ada_ln_w, ada_ln_bias = self.ada_ln_hypernet(slot_set_rep).chunk(2, dim=-1)
        ada_down_proj = self.ada_down_hypernet(slot_set_rep).view(-1, self.slot_dim, self.down_proj_dim)
        ada_up_proj = self.ada_up_hypernet(slot_set_rep).view(-1, self.down_proj_dim, self.slot_dim)

        return HypernetOutput(
            ln_w=ln_w,
            ln_b=ln_bias,
            ada_down_proj=ada_down_proj,
            ada_up_proj=ada_up_proj,
            ada_ln_w=ada_ln_w,
            ada_ln_b=ada_ln_bias,
            use_ada=self.use_ada
        )