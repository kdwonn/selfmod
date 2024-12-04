"""Train a slot attention type model."""
import dataclasses
from typing import Any, Dict, Optional
from functools import partial

import hydra
import hydra_zen
import pytorch_lightning as pl
import omegaconf
import os
import torch
from omegaconf import SI

from ocl.cli import cli_utils
import ocl.cli._config  # noqa: F401
from ocl.combined_model import CombinedModel, CombinedModelAlterOpt, CombinedModelAlterOptAttn
from pytorch_lightning.loggers import WandbLogger
from ocl.cli.cli_utils import create_symlink, get_parent

# --8<-- [start:TrainingConfig]
# Convert dict of callbacks in experiment to list for use with PTL.
CALLBACK_INTERPOLATION = SI("${oc.dict.values:experiment.callbacks}")

TrainerConf = hydra_zen.builds(
    pl.Trainer, callbacks=CALLBACK_INTERPOLATION, zen_partial=False, populate_full_signature=True
)

LoggerConf = hydra_zen.builds(
    WandbLogger, 
    project='ocl-pool', 
    notes=None, 
    tags=None, 
    group=None,
    zen_partial=False, 
    populate_full_signature=True
)

@dataclasses.dataclass
class TrainingConfig:
    """Configuration of a training run.

    For losses, metrics and visualizations it can be of use to utilize the
    [routed][] module as these are simply provided with a dictionary of all
    model inputs and outputs.

    Attributes:
        dataset: The pytorch lightning datamodule that will be used for training
        models: Either a dictionary of [torch.nn.Module][]s which will be interpreted
            as a [Combined][ocl.utils.routing.Combined] model or a [torch.nn.Module][] itself
            that accepts a dictionary as input.
        optimizers: Dictionary of [functools.partial][] wrapped optimizers or
            [OptimizationWrapper][ocl.optimization.OptimizationWrapper] instances
        losses: Dict of callables that return scalar values which will be summed to
            compute a total loss.  Typically should contain [routed][] versions of callables.
        visualizations: Dictionary of [visualizations][ocl.visualizations].  Typically
            should contain [routed][] versions of visualizations.
        trainer: Pytorch lightning trainer
        training_vis_frequency: Number of optimization steps between generation and
            storage of visualizations.
        training_metrics: Dictionary of torchmetrics that should be used to log training progress.
            Typically should contain [routed][] versions of torchmetrics.
        evaluation_metrics: Dictionary of torchmetrics that should be used to log progress on
            evaluation splits of the data.  Typically should contain [routed][] versions of
            Torchmetrics.
        load_checkpoint: Path to checkpoint file that should be loaded prior to starting training.
        seed: Seed used to ensure reproducability.
        experiment: Dictionary with arbitrary additional information.  Useful when building
            configurations as it can be used as central point for a single parameter that might
            influence multiple model components.
    """

    dataset: Any
    models: Any  # When provided with dict wrap in `utils.Combined`, otherwise interpret as model.
    optimizers: Dict[str, Any]
    losses: Dict[str, Any]
    lr_scale: float = 1
    visualizations: Dict[str, Any] = dataclasses.field(default_factory=dict)
    trainer: TrainerConf = dataclasses.field(default_factory=lambda: TrainerConf())
    training_vis_frequency: Optional[int] = None
    training_metrics: Optional[Dict[str, Any]] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None
    load_checkpoint: Optional[str] = None
    seed: Optional[int] = None
    experiment: Dict[str, Any] = dataclasses.field(default_factory=lambda: {"callbacks": {}})
    logger: LoggerConf = dataclasses.field(default_factory=lambda: LoggerConf())
    alter_opt: Optional[str] = None
    alter_interval: int = 3
    alter_warmup_iter: int = 10000


# --8<-- [end:TrainingConfig]


hydra.core.config_store.ConfigStore.instance().store(
    name="training_config",
    node=TrainingConfig,
)


def build_and_register_datamodule_from_config(
    config: TrainingConfig,
    **datamodule_kwargs,
) -> pl.LightningDataModule:
    datamodule = hydra_zen.instantiate(config.dataset, _convert_="all", **datamodule_kwargs)
    return datamodule


def build_model_from_config(
    config: TrainingConfig,
    checkpoint_path: Optional[str] = None,
) -> pl.LightningModule:
    models = hydra_zen.instantiate(config.models, _convert_="all")
    optimizers = hydra_zen.instantiate(config.optimizers, _convert_="all")
    losses = hydra_zen.instantiate(config.losses, _convert_="all")
    visualizations = hydra_zen.instantiate(config.visualizations, _convert_="all")

    training_metrics = hydra_zen.instantiate(config.training_metrics)
    evaluation_metrics = hydra_zen.instantiate(config.evaluation_metrics)

    train_vis_freq = config.training_vis_frequency if config.training_vis_frequency else 100

    # alter_config = {'alter_interval': config.alter_interval, 'alter_warmup_iter': config.alter_warmup_iter}

    # if config.alter_opt is None:
    #     combined = CombinedModel
    # elif config.alter_opt == 'all':
    #     combined = CombinedModelAlterOpt
    # elif config.alter_opt == 'attn':
    #     combined = CombinedModelAlterOptAttn

    combined = CombinedModel

    if checkpoint_path is None:
        model = combined(
            models=models,
            optimizers=optimizers,
            losses=losses,
            visualizations=visualizations,
            training_metrics=training_metrics,
            evaluation_metrics=evaluation_metrics,
            vis_log_frequency=train_vis_freq,
            # **(alter_config if config.alter_opt is not None else {})
        )
    else:
        model = combined.load_from_checkpoint(
            checkpoint_path,
            models=models,
            map_location=torch.device(f'cuda:{config.trainer.devices[0]}'),
            optimizers=optimizers,
            losses=losses,
            visualizations=visualizations,
            training_metrics=training_metrics,
            evaluation_metrics=evaluation_metrics,
            vis_log_frequency=train_vis_freq,
            # **(alter_config if config.alter_opt is not None else {})
        )
    return model


def build_wandblogger_from_config(
        config: TrainingConfig, 
        run_dir: str, 
        dataset_name: str
):
    if config.trainer.logger is False:
        return None
    logger = hydra_zen.instantiate(config.logger, _convert_="all", group=dataset_name)
    config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    config.update({"run_dir": run_dir})
    logger.experiment.config.update(config)
    return logger


@hydra.main(config_name="training_config", config_path="../../configs/", version_base="1.1")
def train(config: TrainingConfig):
    # Set all relevant random seeds. If `config.seed is None`, the function samples a random value.
    # The function takes care of correctly distributing the seed across nodes in multi-node training,
    # and assigns each dataloader worker a different random seed.
    # IMPORTANTLY, we need to take care not to set a custom `worker_init_fn` function on the
    # dataloaders (or take care of worker seeding ourselves).
    pl.seed_everything(config.seed, workers=True)

    datamodule = build_and_register_datamodule_from_config(config)
    model = build_model_from_config(config)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    dataset_name = hydra.core.hydra_config.HydraConfig.get().runtime.choices.dataset
    wandb_logger = build_wandblogger_from_config(config, output_dir, dataset_name)

    root_dir = get_parent(os.path.realpath(__file__), t=3)
    symlink = os.path.join(root_dir, 'outputs/wandb_run', wandb_logger.experiment.name)
    create_symlink(output_dir, symlink)

    callbacks = hydra_zen.instantiate(config.trainer.callbacks, _convert_="all")
    callbacks = callbacks if callbacks else []
    if config.trainer.logger is not False:
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    trainer: pl.Trainer = hydra_zen.instantiate(config.trainer, callbacks=callbacks, logger=wandb_logger, _convert_="all")
    wandb_logger.watch(model, log_freq=500, log="all")

    if config.load_checkpoint:
        # checkpoint_path = hydra.utils.to_absolute_path(config.load_checkpoint)
        checkpoint_path = cli_utils.find_checkpoint(config.load_checkpoint, is_wandb=True)
    else:
        checkpoint_path = None

    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    train()
