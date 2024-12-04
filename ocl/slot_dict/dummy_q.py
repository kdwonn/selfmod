import torch
import torch.nn as nn

class DummyQ(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        return x, torch.zeros_like(x[:, :, 0]).cuda(), torch.zeros(1).cuda()