"""Types used in object centric learning framework."""
import dataclasses
from typing import Dict, Iterable, Optional, Union, Tuple

import torch
from torchtyping import TensorType
from ocl.decoding import PatchReconstructionOutput

# Input data types.
ImageData = TensorType["batch size", "channels", "height", "width"]  # noqa: F821
VideoData = TensorType["batch size", "frames", "channels", "height", "width"]  # noqa: F821
ImageOrVideoData = Union[VideoData, ImageData]  # noqa: F821
TextData = TensorType["batch_size", "max_tokens"]  # noqa: F821

# Feature data types.
CNNImageFeatures = ImageData
TransformerImageFeatures = TensorType[
    "batch_size", "n_spatial_features", "feature_dim"
]  # noqa: F821
ImageFeatures = TransformerImageFeatures
VideoFeatures = TensorType["batch_size", "frames", "n_spatial_features", "feature_dim"]  # noqa: F821
ImageOrVideoFeatures = Union[ImageFeatures, VideoFeatures]
Positions = TensorType["n_spatial_features", "spatial_dims"]  # noqa: F821
PooledFeatures = TensorType["batch_size", "feature_dim"]

# Object feature types.
ObjectFeatures = TensorType["batch_size", "n_objects", "object_dim"]  # noqa: F821
EmptyIndicator = TensorType["batch_size", "n_objects"]  # noqa: F821
ObjectFeatureAttributions = TensorType["batch_size", "n_objects", "n_spatial_features"]  # noqa: F821

# Module output types.
ConditioningOutput = TensorType["batch_size", "n_objects", "object_dim"]  # noqa: F821
"""Output of conditioning modules."""


@dataclasses.dataclass
class FrameFeatures:
    """Features associated with a single frame."""

    features: ImageFeatures
    positions: Positions


@dataclasses.dataclass
class FeatureExtractorOutput:
    """Output of feature extractor."""

    features: ImageOrVideoFeatures
    positions: Positions
    aux_features: Optional[Dict[str, torch.Tensor]] = None
    prompt: Optional[torch.Tensor] = None

    def __iter__(self) -> Iterable[ImageFeatures]:
        """Iterate over features and positions per frame."""
        if self.features.ndim == 4:
            yield FrameFeatures(self.features, self.positions)
        else:
            for frame_features in torch.split(self.features, 1, dim=1):
                yield FrameFeatures(frame_features.squeeze(1), self.positions)


@dataclasses.dataclass
class ScalesIA3:
    q_scale: torch.Tensor
    v_scale: torch.Tensor
    ff_scale: torch.Tensor


@dataclasses.dataclass
class PerceptualGroupingOutput:
    """Output of a perceptual grouping algorithm."""

    objects: ObjectFeatures
    is_empty: Optional[EmptyIndicator] = None  # noqa: F821
    feature_attributions: Optional[ObjectFeatureAttributions] = None  # noqa: F821
    feature_attributions_2nd: Optional[ObjectFeatureAttributions] = None
    feature_attributions_1st: Optional[ObjectFeatureAttributions] = None
    # AbS only outputs
    pool_indices: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.Tensor] = None
    alpha_weights: Optional[ObjectFeatureAttributions] = None
    feedback: Optional[ImageOrVideoData] = None
    feedback_recon: Optional[ImageOrVideoData] = None
    pos_feat: Optional[torch.Tensor] = None
    k_inputs: Optional[torch.Tensor] = None
    v_inputs: Optional[torch.Tensor] = None
    pos_feedback: Optional[torch.Tensor] = None
    pre_attn: Optional[torch.Tensor] = None
    post_attn: Optional[torch.Tensor] = None
    ln_w: Optional[torch.Tensor] = None
    ln_b: Optional[torch.Tensor] = None
    pre_slots: Optional[torch.Tensor] = None
    q_slots: Optional[torch.Tensor] = None
    pool_logit: Optional[torch.Tensor] = None
    gumbel_temp: Optional[torch.Tensor] = None
    mod_weight: Optional[float] = None
    scales: Optional[ScalesIA3] = None

@dataclasses.dataclass
class PerceptualGroupingDisentOutput:
    """Output of a perceptual grouping algorithm."""

    objects: ObjectFeatures
    positions: ObjectFeatures
    is_empty: Optional[EmptyIndicator] = None  # noqa: F821
    feature_attributions: Optional[ObjectFeatureAttributions] = None  # noqa: F821
    rel_pos_grid: Optional[torch.Tensor] = None

@dataclasses.dataclass
class SynthesizerOutput:
    """Output of a synthesizer."""

    feedback: ImageOrVideoData
    reconstruction: Optional[ImageOrVideoData] = None
    alpha_weights: Optional[ObjectFeatureAttributions] = None
    feedback_type: Optional[str] = None


@dataclasses.dataclass
class PerceptualGroupingDecodingOutput:
    """Output of a perceptual grouping with decoding afterward."""

    objects: ObjectFeatures
    object_decoder: PatchReconstructionOutput
    is_empty: Optional[EmptyIndicator] = None  # noqa: F821
    feature_attributions: Optional[ObjectFeatureAttributions] = None  # noqa: F821
    # AbS only outputs
    pool_indices: Optional[torch.Tensor] = None
    top_k_indices: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.Tensor] = None
    alpha_weights: Optional[ObjectFeatureAttributions] = None
    feedback: Optional[ImageOrVideoData] = None
    pos_feat: Optional[torch.Tensor] = None
    k_inputs: Optional[torch.Tensor] = None
    v_inputs: Optional[torch.Tensor] = None
    pos_feedback: Optional[torch.Tensor] = None
    pre_attn: Optional[torch.Tensor] = None
    post_attn: Optional[torch.Tensor] = None
    pre_slots: Optional[torch.Tensor] = None
    q_slots: Optional[torch.Tensor] = None


@dataclasses.dataclass
class PerceptualGroupingModDecOutput:
    """Output of a perceptual grouping with decoding afterward."""

    objects: ObjectFeatures
    dec_to_train: PatchReconstructionOutput
    pre_object_decoder: PatchReconstructionOutput
    object_decoder: PatchReconstructionOutput
    is_empty: Optional[EmptyIndicator] = None  # noqa: F821
    feature_attributions: Optional[ObjectFeatureAttributions] = None  # noqa: F821
    # AbS only outputs
    pool_indices: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.Tensor] = None
    alpha_weights: Optional[ObjectFeatureAttributions] = None
    prompt: Optional[ImageOrVideoData] = None
    prompted_feature: Optional[ImageOrVideoData] = None
    feature: Optional[ImageOrVideoData] = None
    k_inputs: Optional[torch.Tensor] = None
    v_inputs: Optional[torch.Tensor] = None
    pre_attn: Optional[torch.Tensor] = None
    post_attn: Optional[torch.Tensor] = None
    pre_slots: Optional[torch.Tensor] = None
    q_slots: Optional[torch.Tensor] = None
    pool_logit: Optional[torch.Tensor] = None
    gumbel_temp: Optional[torch.Tensor] = None


@dataclasses.dataclass
class PerceptualGroupingAdaGNOutput:
    objects: ObjectFeatures
    is_empty: Optional[EmptyIndicator] = None  # noqa: F821
    feature_attributions: Optional[ObjectFeatureAttributions] = None  # noqa: F821
    # AbS only outputs
    pool_indices: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.Tensor] = None
    feature: Optional[ImageOrVideoData] = None
    ada_feature: Optional[ImageOrVideoData] = None
    pre_attn: Optional[torch.Tensor] = None
    pre_slots: Optional[torch.Tensor] = None
    q_slots: Optional[torch.Tensor] = None


@dataclasses.dataclass
class PerceptualGroupingAdaGNDecOutput:
    objects: ObjectFeatures
    object_decoder: PatchReconstructionOutput
    pre_object_decoder: PatchReconstructionOutput
    is_empty: Optional[EmptyIndicator] = None  # noqa: F821
    feature_attributions: Optional[ObjectFeatureAttributions] = None  # noqa: F821
    # AbS only outputs
    pool_indices: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.Tensor] = None
    feature: Optional[ImageOrVideoData] = None
    ada_feature: Optional[ImageOrVideoData] = None
    pre_attn: Optional[torch.Tensor] = None
    pre_slots: Optional[torch.Tensor] = None
    q_slots: Optional[torch.Tensor] = None