from fvcore.nn import FlopCountAnalysis
import torch

import dataclasses
import json
import logging
import math
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import hydra
import hydra_zen
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict

import ocl.cli._config  # noqa: F401
from ocl.cli import cli_utils, eval_utils, train

from calflops import calculate_flops
from ptflops import get_model_complexity_info

@hydra.main(config_name="self_mod_arch", config_path="../../", version_base="1.1")
def eval(config):
    model = hydra.utils.instantiate(config.models.feedback_model).to("cuda:2")
    model.eval()
    input = (
        torch.zeros((1, 3, 224, 224)).to(
            device=model.perceptual_grouping.slot_attention.to_q.weight.device,
            dtype=model.perceptual_grouping.slot_attention.to_q.weight.dtype
        ), 1
    )
    # flop = FlopCountAnalysis(
    #     model, 
    #     input
    # )
    flop = calculate_flops(model, kwargs={"image": input[0]})
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    eval()