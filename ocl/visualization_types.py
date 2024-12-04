"""Classes for handling different types of visualizations."""
import dataclasses
from typing import Any, List, Optional, Union

import matplotlib.pyplot
import torch
import wandb
import numpy as np
from torchtyping import TensorType
from numpy import array
from einops import rearrange


class SummaryWriter:
    """Placeholder class for SummaryWriter.

    Emulates interface of `torch.utils.tensorboard.SummaryWriter`.
    """

    def add_figure(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_images(self, *args, **kwargs):
        pass

    def add_video(self, *args, **kwargs):
        pass

    def add_embedding(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass


class WandbLogger:
    """Placeholder class for WandbLogger.

    Emulates interface of `pytorch_lightning.loggers.WandbLogger`.
    """ 
    def log_image(self, *args, **kwargs):
        pass

    def log_text(self, *args, **kwargs):
        pass

    def log_table(self, *args, **kwargs):
        pass


def wandb_image_fn(x, caption=None):
    x = rearrange(x, '... d h w -> ... h w d').clone().cpu().data.numpy()
    if len(x.shape) == 4:
        ret = [wandb.Image(i, caption=caption[i]) for i in np.split(x, x.shape[0], axis=0)]
    elif len(x.shape) == 3:
        ret = [wandb.Image(x, caption=caption)]
    else:
        raise ValueError(f"Invalid iamge tensor shape {x.shape}")
    return ret


def dataclass_to_dict(d):
    return {field.name: getattr(d, field.name) for field in dataclasses.fields(d)}


@dataclasses.dataclass
class Visualization:
    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        pass

    def log_to_wandb(self, wandb_logger: WandbLogger, key: str, global_step: int):
        pass

    def get_wandb_log_dict(self, key:str):
        pass


@dataclasses.dataclass
class Figure(Visualization):
    """Matplotlib figure."""

    figure: matplotlib.pyplot.figure
    close: bool = True

    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        experiment.add_figure(**dataclass_to_dict(self), tag=tag, global_step=global_step)

    def log_to_wandb(self, wandb_logger: WandbLogger, key: str, global_step: int):
        wandb_logger.log_image(image=wandb_image_fn(self.figure), key=key, step=global_step)

    def get_wandb_log_dict(self, key: str):
        return {key: wandb_image_fn(self.figure)}


@dataclasses.dataclass
class Image(Visualization):
    """Single image."""

    img_tensor: torch.Tensor
    dataformats: str = "CHW"
    caption: Optional[str] = None

    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        experiment.add_image(**dataclass_to_dict(self), tag=tag, global_step=global_step)

    def log_to_wandb(self, wandb_logger: WandbLogger, key: str, global_step: int):
        to_log = {
            "images": wandb_image_fn(self.img_tensor),
            "key": key,
            "step": global_step,
        }
        to_log.update({"caption": self.caption} if self.caption is not None else {})
        wandb_logger.log_image(**to_log)

    def get_wandb_log_dict(self, key: str):
        return {key: wandb_image_fn(self.img_tensor, caption=self.caption)}


@dataclasses.dataclass
class Images(Visualization):
    """Batch of images."""

    img_tensor: torch.Tensor
    dataformats: str = "NCHW"
    caption: Optional[str] = None

    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        experiment.add_images(**dataclass_to_dict(self), tag=tag, global_step=global_step)

    def log_to_wandb(self, wandb_logger: WandbLogger, key: str, global_step: int):
        to_log = {
            "images": wandb_image_fn(self.img_tensor),
            "key": key,
            "step": global_step,
        }
        to_log.update({"caption": self.caption} if self.caption is not None else {})
        wandb_logger.log_image(**to_log)

    def get_wandb_log_dict(self, key: str):
        return {key: wandb_image_fn(self.img_tensor, caption=self.caption[0])}


@dataclasses.dataclass
class Video(Visualization):
    """Batch of videos."""

    vid_tensor: TensorType["batch_size", "frames", "channels", "height", "width"]  # noqa: F821
    fps: Union[int, float] = 4

    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        experiment.add_video(**dataclass_to_dict(self), tag=tag, global_step=global_step)

    def log_to_wandb(self, wandb_logger: WandbLogger, key: str, global_step: int):
        wandb_logger.experiment.log({key: wandb.Video(self.vid_tensor), "trainer/global_step": global_step}, step=global_step)

    def get_wandb_log_dict(self, key: str):
        raise NotImplementedError


@dataclasses.dataclass
class Embedding(Visualization):
    """Batch of embeddings."""

    mat: TensorType["batch_size", "feature_dim"]  # noqa: F821
    metadata: Optional[List[Any]] = None
    label_img: Optional[TensorType["batch_size", "channels", "height", "width"]] = None  # noqa: F821
    metadata_header: Optional[List[str]] = None

    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        experiment.add_embedding(**dataclass_to_dict(self), tag=tag, global_step=global_step)

    def log_to_wandb(self, wandb_logger: WandbLogger, key: str, global_step: int):
        wandb_logger.experiment.log({key: self.mat, "trainer/global_step": global_step}, step=global_step)

    def get_wandb_log_dict(self, key: str):
        raise NotImplementedError
    

@dataclasses.dataclass
class Histogram(Visualization):
    hist: array

    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        experiment.add_histogram(experiment, tag, global_step)

    def log_to_wandb(self, wandb_logger: WandbLogger, key: str, global_step: int):
        wandb_logger.experiment.log({key: wandb.Histogram(np_histogram=self.hist), "trainer/global_step": global_step}, step=global_step)

    def get_wandb_log_dict(self, key: str):
        return {key: wandb.Histogram(np_histogram=self.hist)}
