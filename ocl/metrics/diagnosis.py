"""Metrics used for diagnosis."""
from typing import Any
import torch
import torch.nn.functional as F
import torchmetrics
import torch.linalg as LA


class TensorStatistic(torchmetrics.Metric):
    """Metric that computes summary statistic of tensors for logging purposes.

    First dimension of tensor is assumed to be batch dimension. Other dimensions are reduced to a
    scalar by the chosen reduction approach (sum or mean).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("sum", "mean", "var"):
            raise ValueError(f"Unknown reduction {reduction}")
        self.reduction = reduction
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, tensor: torch.Tensor):
        tensor = torch.atleast_2d(tensor).flatten(1, -1).to(dtype=torch.float64)

        if self.reduction == "mean":
            tensor = torch.mean(tensor, dim=1)
        elif self.reduction == "sum":
            tensor = torch.sum(tensor, dim=1)
        elif self.reduction == "var":
            tensor = torch.var(tensor, dim=1)

        self.values += tensor.sum()
        self.total += len(tensor)

    def compute(self) -> torch.Tensor:
        return self.values / self.total


class TensorNormStatistic(torchmetrics.Metric):
    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.dim = dim
        self.add_state("values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, tensor: torch.Tensor) -> None:
        tensor = torch.atleast_2d(tensor).flatten(1, -1).to(dtype=torch.float64)
        tensor = LA.vector_norm(tensor, dim=self.dim)
        self.values += tensor.sum()
        self.total += len(tensor)

    def compute(self) -> torch.Tensor:
        return self.values / self.total
    

class TensorNormRatioStatistic(torchmetrics.Metric):
    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.dim = dim
        self.add_state("values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
        tensor1 = torch.atleast_2d(tensor1).flatten(1, -1).to(dtype=torch.float64)
        tensor2 = torch.atleast_2d(tensor2).flatten(1, -1).to(dtype=torch.float64)
        tensor1 = LA.vector_norm(tensor1, dim=self.dim)
        tensor2 = LA.vector_norm(tensor2, dim=self.dim)
        tensor = tensor1 / tensor2
        self.values += tensor.sum()
        self.total += len(tensor)

    def compute(self) -> torch.Tensor:
        return self.values / self.total
    

class TensorDistStatistic(torchmetrics.Metric):
    def __init__(self, dist):
        super().__init__()
        if dist not in ("l2", "l1", "cos"):
            raise ValueError(f"Unknown dist {dist}")
        self.dist = dist
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        if self.dist == "l2":
            self.dist_fn = lambda t1, t2: LA.vector_norm(t1 - t2, dim=-1)
        elif self.dist == "l1":
            self.dist_fn = lambda t1, t2: LA.vector_norm(t1 - t2, ord=1, dim=-1)
        elif self.dist == "cos":
            self.dist_fn = lambda t1, t2: 1 - F.cosine_similarity(t1, t2, dim=-1)

    def update(self, tensor1, tensor2):
        tensor1 = torch.atleast_2d(tensor1).to(dtype=torch.float64)
        tensor2 = torch.atleast_2d(tensor2).to(dtype=torch.float64)
        tensor = self.dist_fn(tensor1, tensor2)
        self.values += tensor.sum()
        self.total += len(tensor.flatten())

    def compute(self) -> torch.Tensor:
        return self.values / self.total