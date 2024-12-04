import torch
import torch.nn as nn
from ocl.slot_dict.vq import VectorQuantize
from einops import rearrange

class BlockVectorQuantize(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.num_blocks = kwargs.pop('num_blocks')
        self.vqs = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.vqs.append(VectorQuantize(**kwargs))

    def forward(self, x):
        assert x.shape[-1] % self.num_blocks == 0
        x = rearrange(x, 'b n (k d) -> b n k d', k=self.num_blocks)

        codes, inds, commits = [], [], []
        for i in range(self.num_blocks):
            code, ind, commit = self.vqs[i](x[:, :, i, :])
            codes.append(code)
            inds.append(ind.unsqueeze(-1))
            commits.append(commit)

        codes = torch.cat(codes, dim=-1)
        inds = torch.cat(inds, dim=-1)
        commits = torch.cat(commits, dim=-1)

        return codes, inds, commits