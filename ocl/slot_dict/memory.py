# implementation of the compressive memory from the paper "Memory augmented Dense Prediction Coding for Video Representation Learning"
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

class MemDPC(nn.Module):
    def __init__(
            self,
            in_dim,
            code_dim,
            codebook_size,
        ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.code_dim = code_dim
        self.codebook_size = codebook_size

        self.use_squeezed_bottleneck = self.code_dim == self.in_dim
        assert self.code_dim <= self.in_dim

        self.unsqueezer = nn.Linear(self.code_dim, self.in_dim) if not self.use_squeezed_bottleneck else nn.Identity()

        self.scoring_func = nn.Linear(self.in_dim, self.codebook_size)
        self.prob_dropout = nn.Dropout(0.1)

        self.codebook = nn.Parameter(torch.randn(codebook_size, code_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.codebook)
        nn.init.xavier_uniform_(self.scoring_func.weight)
        nn.init.zeros_(self.scoring_func.bias)
        if self.use_squeezed_bottleneck:
            nn.init.zeros_(self.unsqueezer.bias)
            nn.init.xavier_uniform_(self.unsqueezer.weight)

    def forward(self, x):
        assert len(x.shape) == 3
        b, n, d = x.shape

        prob = F.softmax(self.scoring_func(x), dim=-1)
        prob = self.prob_dropout(prob)
        code = einsum(self.codebook, prob, 'c d, b n c -> b n d')
        unsqueezed_code = self.unsqueezer(code)

        commit_loss = F.mse_loss(unsqueezed_code, x.detach())

        inds = torch.argmax(prob, dim=-1)

        return unsqueezed_code, inds, commit_loss
    

class DropClassifier(nn.Module):
    def __init__(
            self,
            in_dim,
            codebook_size,
            dropout=False
        ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.codebook_size = codebook_size

        self.scoring_func = nn.Linear(self.in_dim, self.codebook_size)
        self.prob_dropout = nn.Dropout(0.1) if dropout else nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.scoring_func.weight)
        nn.init.zeros_(self.scoring_func.bias)

    def forward(self, x):
        assert len(x.shape) == 3

        prob = F.softmax(self.scoring_func(x), dim=-1)
        prob = self.prob_dropout(prob)

        return None, torch.argmax(prob, dim=-1), torch.zeros((1), device=x.device), prob