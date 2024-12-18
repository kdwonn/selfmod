"""Package for metrics.

The implemetation of metrics are grouped into submodules according to their
datatype and use

 - [ocl.metrics.bbox][]: Metrics for bounding boxes
 - [ocl.metrics.masks][]: Metrics for masks
 - [ocl.metrics.tracking][]: Metrics for multiple object tracking.
 - [ocl.metrics.diagnosis][]: Metrics for diagnosing model training.
 - [ocl.metrics.dataset][]: Metrics that are computed on the whole dataset.

"""

from ocl.metrics.bbox import BboxCorLocMetric, BboxRecallMetric, UnsupervisedBboxIoUMetric
from ocl.metrics.dataset import DatasetSemanticMaskIoUMetric, SklearnClustering
from ocl.metrics.diagnosis import TensorStatistic, TensorNormRatioStatistic, TensorNormStatistic, TensorDistStatistic
from ocl.metrics.masks import (
    ARIMetric,
    AverageBestOverlapMetric,
    BestOverlapObjectRecoveryMetric,
    MaskCorLocMetric,
    PatchARIMetric,
    UnsupervisedMaskIoUMetric,
    BestOverlapPixelPrecisionMetric,
    BestOverlapPixelRecallMetric
)
from ocl.metrics.tracking import MOTMetric
from ocl.metrics.pool import UniqueRatio, CodebookUsage, Perplexity, CodebookHistogram

__all__ = [
    "UnsupervisedBboxIoUMetric",
    "BboxCorLocMetric",
    "BboxRecallMetric",
    "DatasetSemanticMaskIoUMetric",
    "SklearnClustering",
    "TensorStatistic",
    "ARIMetric",
    "PatchARIMetric",
    "UnsupervisedMaskIoUMetric",
    "MaskCorLocMetric",
    "AverageBestOverlapMetric",
    "BestOverlapObjectRecoveryMetric",
    "MOTMetric",
    "UniqueRatio",
    "CodebookUsage",
    "Perplexity",
    "BestOverlapPixelPrecisionMetric",
    "BestOverlapPixelRecallMetric"
]
