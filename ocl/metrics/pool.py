from typing import Any
import numpy as np
import torch
import torchmetrics
import torch.nn.functional as F

from scipy.stats import entropy
import wandb


class Perplexity(torchmetrics.Metric):
    def __init__(self, pool_size: int, base:int) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.base = base
        self.add_state("density", default=torch.zeros((self.pool_size)), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros((1)), dist_reduce_fx="sum")

    def update(self, indices: torch.Tensor) -> None:
        indices = indices.cpu().data.numpy()
        density, _ = np.histogram(indices, bins=self.pool_size, range=(0, self.pool_size), density=True)
        self.density += torch.tensor(density).to(device=self.device)
        self.count += 1

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return 0
        else:
            density = self.density.cpu().data.numpy()
            count = self.count.cpu().data.numpy()
            perplexity = 2**(entropy((density/count), base=self.base))
            return torch.tensor(perplexity).to(device=self.device)
        

class Scalar(torchmetrics.Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("val", default=torch.zeros((1)), dist_reduce_fx="mean")

    def update(self, value) -> None:
        self.val = torch.tensor(value)

    def compute(self):
        return self.val


class Entropy(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_entropy", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor):
        # Apply softmax to convert logits into probabilities
        probabilities = F.softmax(preds, dim=-1)
        # Calculate log probabilities
        log_probs = torch.log(probabilities)
        # Calculate entropy for the batch
        entropy_values = -torch.sum(probabilities * log_probs, dim=-1)
        batch_entropy = torch.mean(entropy_values)  # Average entropy for the batch
        
        # Accumulate the average entropy and count the batch
        self.total_entropy += batch_entropy
        self.total_batches += 1

    def compute(self):
        # Compute the average entropy across all batches
        if self.total_batches == 0:
            return torch.tensor(0.)  # To handle division by zero gracefully
        return self.total_entropy / self.total_batches

class UniqueRatio(torchmetrics.Metric):
    def __init__(self, slot_num: int) -> None:
        super().__init__()
        self.slot_num = slot_num
        self.add_state("unique_ratio", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def _row_unique_count(self, a):
        a = a.astype(str)
        if len(a.shape) == 3:
            # For multi-head codebook, we need to join the indices of each head into unique strings
            a = np.apply_along_axis(lambda x: ','.join(x), -1, a)
        unique = np.sort(a)
        duplicates = unique[:,  1:] == unique[:, :-1]
        unique[:, 1:][duplicates] = ''
        unique = (unique != '').astype(int)
        return unique.sum(axis=1)

    def update(self, indices: torch.Tensor) -> None:
        indices = indices.cpu().data.numpy()
        unique_count = self._row_unique_count(indices)
        self.unique_ratio += unique_count.mean() / self.slot_num
        self.count += 1

    def compute(self) -> torch.Tensor:
        return torch.tensor(self.unique_ratio / self.count)
    

class CodebookUsage(torchmetrics.Metric):
    def __init__(self, pool_size: int) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.add_state("codebook_usage", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def _total_unique_count(self, a):
        # Count the number of unique elements in a 2D array
        # FIXME Needs fixes for multi-head attention without codebook sharing,
        # where the sane index can be used multiple times. 
        unique = np.sort(a.flatten())
        duplicates = unique[1:] == unique[:-1]
        unique[1:][duplicates] = 0
        unique = (unique != 0).astype(int)
        return unique.sum()

    def update(self, indices: torch.Tensor) -> None:
        indices = indices.cpu().data.numpy()
        total_unique_count = self._total_unique_count(indices)
        self.codebook_usage += total_unique_count / self.pool_size
        self.count += 1

    def compute(self) -> torch.Tensor:
        return torch.tensor(self.codebook_usage / self.count)
    

class CodebookHistogram(torchmetrics.Metric):
    def __init__(self, num_bins=10, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("indices", default=torch.tensor([]), dist_reduce_fx="cat")
        self.num_bins = num_bins

    def update(self, indices: torch.Tensor):
        self.indices = torch.cat((self.indices, indices))

    def compute(self):
        # Create the histogram
        histogram = np.histogram(self.indices.cpu().data.numpy(), bins=range(self.num_bins + 1))
        wandb_histogram = wandb.Histogram(np_histogram=histogram, num_bins=self.num_bins)
        #  dumb hack to log histogram to wandb, metrics should return a tensor
        wandb.log({'val/pool_hist': wandb_histogram})
        return torch.ones((1))
