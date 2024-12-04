import glob
import os
import wandb

from hydra.core.hydra_config import HydraConfig


def get_commandline_config_path():
    """Get the path of a config path specified on the command line."""
    hydra_cfg = HydraConfig.get()
    config_sources = hydra_cfg.runtime.config_sources
    config_path = None
    for source in config_sources:
        if source.schema == "file" and source.provider == "command-line":
            config_path = source.path
            break
    return config_path


def find_checkpoint(path, is_wandb=False):
    """Find checkpoint in output path of previous run."""
    if is_wandb:
        #  wandb version: workspace name, run-id, checkpoints, checkpoint name
        checkpoints = glob.glob(
            os.path.join(path, "*", "*", "checkpoints", "*.ckpt")
        )
    else:
        checkpoints = glob.glob(
            os.path.join(path, "lightning_logs", "version_*", "checkpoints", "*.ckpt")
        )
    checkpoints.sort()
    # Return the last checkpoint.
    # TODO (hornmax): If more than one checkpoint is stored this might not lead to the most recent
    # checkpoint being loaded. Generally, I think this is ok as we still allow people to set the
    # checkpoint manually.
    return checkpoints[-1]


def create_symlink(target_dir, link_name):
    """
    Create a symbolic link pointing to the target directory

    :param target_dir: The path of the directory to which the symlink will point.
    :param link_name: The path of the symlink to create.
    """
    try:      
        # Create a symbolic link
        os.symlink(target_dir, link_name)
        print(f"Symbolic link created: {link_name} -> {target_dir}")
    except OSError as e:
        print(f"Error: {e}")


def get_parent(x, t=1):
    if t <= 1:
        return os.path.dirname(x)
    else:
        return get_parent(os.path.dirname(x), t-1)