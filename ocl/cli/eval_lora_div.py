#!/usr/bin/env python
"""Evaluate a trained model for object discovery by clustering object representations.

Given a set of images, each with a set of ground truth masks and a set of object masks and
representations, we perform the following steps:
    1) Assign each object a cluster id by clustering the corresponding representations over all
    images.
    2) Merge object masks with the same cluster id on the same image to form a semantic mask.
    3) Compute IoU between masks of predicted clusters and ground truth classes over all images.
    4) Assign clusters to classes based on the IoU and a matching strategy.
"""
import dataclasses
import enum
import json
import logging
import os
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, List, Optional

import hydra
import hydra_zen
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
import torchvision
import math

from ocl import feature_extractors, metrics
from ocl.cli import cli_utils, eval_utils, train

logger = logging.getLogger("eval_cluster_metrics")


class RepresentationType(enum.Enum):
    NONE = enum.auto()
    SLOTS = enum.auto()
    MASK_WEIGHTED_FEATURES = enum.auto()
    CLS_ON_MASKED_INPUT = enum.auto()


# --8<-- [start:EvaluationClusteringConfig]
@dataclasses.dataclass
class EvaluationClusteringConfig:
    """Configuration for evaluation."""

    # Path to training configuration file or configuration dir. If dir, train_config_name
    # needs to be set as well.
    train_config_path: str

    # Number of classes. Note that on COCO, this should be one larger than the maximum class ID that
    # can appear, which does not correspond to the real number of classes.
    n_classes: int
    # Clustering methods to get cluster ID per object by clustering representations
    # This only supports clustering metrics.
    clusterings: Optional[Dict[str, Any]] = None
    # Paths for model outputs to get cluster ID per object
    model_clusterings: Optional[Dict[str, str]] = None

    train_config_overrides: Optional[List[str]] = None
    train_config_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    output_dir: Optional[str] = None
    report_filename: str = "clustering_metrics.json"

    batch_size: int = 25
    class_discovery_threshold: float = 0.02
    use_mask_threshold: bool = False
    mask_threshold: float = 0.5
    ignore_background: bool = False
    use_unmatched_as_background: bool = False
    use_ignore_masks: bool = False
    n_min_mask_pixels: int = 1  # Minimum number of pixels a mask must occupy to be considered valid
    n_min_max_mask_values: float = 1e-4  # Mask must have at least one value above threshold

    # Type of representation to use for clustering.
    representation_type: RepresentationType = RepresentationType.SLOTS

    # Setting this allows to add modules to the model that are executed during evaluation
    modules: Optional[Dict[str, Any]] = None
    # Setting this allows to evaluate on a different dataset than the model was trained on
    dataset: Optional[Any] = None

    # Path to slot representations
    slots_path: str = "perceptual_grouping.objects"
    # Path to feature representations
    features_path: str = "feature_extractor.features"
    # Path to slot masks, image shaped
    masks_path: str = "perceptual_grouping.alpha_weights"
    # Path to slot masks, but flattened to match the size of features
    masks_flat_path: str = "object_decoder.masks"
    # Path under which representation to cluster is stored
    cluster_representation_path: str = "representation"
    # Path under which empty slot mask is stored
    # empty_slots_path: str = "empty_slots"
    masked_image_path: str = "masked_image"

    class_name_by_category_id: Optional[Dict[int, str]] = None

    denormalization: Any = None
    image_path: str = "input.image"
    code_indices_path: str = 'perceptual_grouping.pool_indices'


# --8<-- [end:EvaluationClusteringConfig]


@dataclasses.dataclass
class Results:
    iou_per_class: np.ndarray
    accuracy: np.ndarray
    empty_classes: np.ndarray
    n_classes: int
    n_clusters: int
    has_background: bool
    matching: str
    clustering: Optional[Any] = None
    model_clustering: Optional[str] = None

    def mean_iou(self) -> float:
        return np.mean(self.iou_per_class[~self.empty_classes])

    def num_discovered(self, threshold) -> int:
        iou_non_empty = self.iou_per_class[~self.empty_classes]
        return len(iou_non_empty[iou_non_empty > threshold])

    def mean_iou_discovered(self, threshold) -> float:
        iou_non_empty = self.iou_per_class[~self.empty_classes]
        iou_discovered = iou_non_empty[iou_non_empty > threshold]
        if len(iou_discovered) > 0:
            return np.mean(iou_discovered)
        else:
            return 0.0

    def mean_iou_without_bg(self) -> float:
        if self.has_background:
            return np.mean(self.iou_per_class[1:][~self.empty_classes[1:]])
        else:
            return self.mean_iou()


hydra.core.config_store.ConfigStore.instance().store(
    name="evaluation_clustering_config",
    node=EvaluationClusteringConfig,
)

def get_transforms(image_path, mask_path, masked_image_path, denormalization):
    #  performed batchwise
    def get_masked_image(data):
        denormed = denormalization(data[image_path].cpu())
        masks = rearrange(data[mask_path], 'b o h w -> b o () h w').cpu()
        image = rearrange(denormed, 'b c h w -> b () c h w')
        data[masked_image_path] = masks * image + (1 - masks)
        return data
    return get_masked_image


def get_data_from_model(model, datamodule, input_paths, output_paths, devices, transform=None):
    data_extractor = eval_utils.ExtractDataFromPredictions(input_paths, output_paths, transform)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=devices,
        callbacks=[data_extractor],
        logger=False,
    )
    trainer.predict(model, datamodule.val_dataloader(), return_predictions=False)
    outputs = data_extractor.outputs

    return outputs


def get_cluster_ids_from_clustering(outputs, clustering, cluster_repr_path):
    objects = torch.cat(outputs[cluster_repr_path])
    objects_flat = objects.flatten(0, 1)

    object_ids = clustering.fit_predict(objects_flat)

    object_ids = object_ids.unflatten(0, objects.shape[:2])
    object_ids = object_ids.split([len(obj) for obj in outputs[cluster_repr_path]])
    return object_ids



@hydra.main(
    config_name="evaluation_clustering_config", config_path="../../configs", version_base="1.1"
)
def evaluate(config: EvaluationClusteringConfig):
    config.train_config_path = hydra.utils.to_absolute_path(config.train_config_path)
    if config.train_config_path.endswith(".yaml"):
        config_dir, config_name = os.path.split(config.train_config_path)
    else:
        config_dir, config_name = config.train_config_path, "config.yaml"
        # config_dir, config_name = config.train_config_path, config.train_config_name

    if not os.path.exists(config_dir):
        raise ValueError(f"Inferred config dir at {config_dir} does not exist.")

    if config.checkpoint_path is None:
        try:
            run_dir = os.path.dirname(config_dir)
            checkpoint_path = cli_utils.find_checkpoint(run_dir, is_wandb=True)
            config.checkpoint_path = checkpoint_path
            logger.info(f"Automatically derived checkpoint path: {checkpoint_path}")
        except (TypeError, IndexError):
            raise ValueError(
                "Unable to automatically derive checkpoint from command line provided config file "
                "path. You can manually specify a checkpoint using the `checkpoint_path` argument."
            )
    else:
        config.checkpoint_path = hydra.utils.to_absolute_path(config.checkpoint_path)
        if not os.path.exists(config.checkpoint_path):
            raise ValueError(f"Checkpoint at {config.checkpoint_path} does not exist.")

    if config.output_dir is None:
        config.output_dir = run_dir
        if not os.path.exists(config.output_dir):
            os.mkdir(config.output_dir)
        logger.info(f"Using {config.output_dir} as output directory.")

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=config_dir):
        overrides = config.train_config_overrides if config.train_config_overrides else []
        train_config = hydra.compose(os.path.splitext(config_name)[0], overrides=overrides)

        with open_dict(train_config):
            train_config.dataset.batch_size = config.batch_size

        datamodule, model = eval_utils.build_from_train_config(train_config, config.checkpoint_path)

    if config.modules is not None:
        modules = hydra_zen.instantiate(config.modules, _convert_="all")
        for key, module in modules.items():
            model.models[key] = module

    if config.dataset is not None:
        datamodule = train.build_and_register_datamodule_from_config(
            config,
            batch_size=train_config.dataset.batch_size,
            eval_batch_size=config.batch_size,
        )

    config.clusterings = config.clusterings if config.clusterings else {}
    clusterings = hydra_zen.instantiate(config.clusterings)

    denormalization = hydra_zen.instantiate(config.denormalization)
    transform = get_transforms(config.image_path, config.masks_path, config.masked_image_path, denormalization)

    pretransform_paths = [
        config.masks_path,
        config.image_path,
        config.code_indices_path
    ]
    output_paths = [
        config.masked_image_path,
        config.code_indices_path
    ]

    from einops import einsum
    
    k_lora_divs = []
    v_lora_divs = []
    q_lora_divs = []
    f1_lora_divs = []
    f2_lora_divs = []

    def get_hook(divs):
        def lora_div_hook(module, i, o):
            down_slotss = einsum(i[0], module.down_proj_values, 'b k d, n d r -> b k n r')
            out_slotss = einsum(down_slotss, module.up_proj_values, 'b k n r, n r d -> b k n d')
            cdist = torch.cdist(out_slotss.flatten(0, 1), out_slotss.flatten(0, 1))
            divs.append(cdist.mean())
            return
        return lora_div_hook
    
    model.models.perceptual_grouping.slot_attention.k_lora.register_forward_hook(get_hook(k_lora_divs))
    model.models.perceptual_grouping.slot_attention.v_lora.register_forward_hook(get_hook(v_lora_divs))
    model.models.perceptual_grouping.slot_attention.q_lora.register_forward_hook(get_hook(q_lora_divs))
    model.models.perceptual_grouping.slot_attention.f_lora1.register_forward_hook(get_hook(f1_lora_divs))
    model.models.perceptual_grouping.slot_attention.f_lora2.register_forward_hook(get_hook(f2_lora_divs))
    
    outputs = get_data_from_model(model, datamodule, pretransform_paths, output_paths, config.trainer.devices, transform)

    masked_image = torch.cat(outputs[config.masked_image_path])
    indices = torch.cat(outputs[config.code_indices_path])
    output_dir =os.path.join(config.output_dir, "codebook")
    os.makedirs(output_dir, exist_ok=False)
    
    codebook_size = train_config.models.perceptual_grouping.pool.codebook_size
    masked_image_per_codebook = [
        masked_image[indices == i] for i in range(codebook_size)
    ]
    size_per_code = [len(code) for code in masked_image_per_codebook]

    for idx in range(codebook_size):
        if len(masked_image_per_codebook[idx]) <= 0:
            continue
        torchvision.utils.save_image(
            masked_image_per_codebook[idx][:20],
            os.path.join(output_dir, f"code_{idx}_size_{size_per_code[idx]}.png"),
            nrow=5,
            padding=2,
            normalize=True,
            range=(0, 1),
        )

if __name__ == "__main__":
    evaluate()
