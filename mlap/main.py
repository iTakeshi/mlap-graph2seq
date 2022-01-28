from datetime import datetime
import itertools
import os
from pathlib import Path
import re
import subprocess
from typing import List, Optional, Tuple

import numpy as np
import torch as th
from torch import cuda, optim
from torch.backends import cudnn
from torch.nn import functional as F

from mlap.nn.MLAP import MLAPBase, MLAPWeighted
from mlap.tasks.base import Task
from mlap.tasks.ogb import OGBCodeTask
from mlap.util import log


DEVICE = th.device("cuda:0" if cuda.is_available() else "cpu")
TIMESTR = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
GITHASH = os.environ.get("GITHASH") or subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()
MODEL_REGEX = re.compile(r".*models/(?P<dataset>[^\\]+)/[^\\]+/(?P<arch>gin-[^_]+)_(?P<norm>[^_]+)_d(?P<dim>\d+)_l(?P<depth>\d+)_s(?P<seed>\d+)_e\d+")

if cudnn.enabled:
    cudnn.deterministic = True
    cudnn.benchmark = False


def build_pathname(
        *,
        prefix: bool=False,
        arch: Optional[str]=None,
        norm: Optional[str]=None,
        residual: bool=False,
        dim_feat: Optional[int]=None,
        depth: Optional[int]=None,
        seed: Optional[int]=None,
        batch_size: Optional[int]=None,
        learning_rate: Optional[Tuple[float, int, float]]=None,
        suffix: Optional[str]=None
) -> str:

    name = ""

    if prefix:
        name = f"{TIMESTR}_{GITHASH}"
    if arch:
        name += f"_{arch}"
    if norm:
        name += f"_{norm}"
    if residual:
        name += f"_res"
    if dim_feat:
        name += f"_d{dim_feat}"
    if depth:
        name += f"_l{depth}"
    if seed is not None:
        name += f"_s{seed}"
    if batch_size is not None:
        name += f"_b{batch_size}"
    if learning_rate is not None:
        name += f"_r{learning_rate[0]}_{learning_rate[1]}_{learning_rate[2]}"
    if suffix:
        name += suffix

    if name.startswith("_"):
        name = name[1:]

    return name


def run_training(
        args_str: str,
        task: Task,
        arch: str,
        norm: str,
        residual: bool,
        dim_feat: int,
        depth: int,
        seed: int,
        epochs: int,
        batch_size: int,
        initial_lr: float,
        lr_interval: int,
        lr_scale: float,
        *,
        save: bool=True,
):

    task.build_model(arch, norm, residual, dim_feat, depth)
    task.set_seed(seed)

    log_path = task.log_dir / build_pathname(prefix=True, arch=arch, norm=norm, residual=residual, dim_feat=dim_feat, depth=depth, seed=seed, batch_size=batch_size, learning_rate=(initial_lr, lr_interval, lr_scale), suffix="_train.log")
    save_dir = task.model_dir / build_pathname(prefix=True, arch=arch, norm=norm, residual=residual, dim_feat=dim_feat, depth=depth, seed=seed, batch_size=batch_size, learning_rate=(initial_lr, lr_interval, lr_scale))
    save_dir.mkdir()
    save_name = build_pathname(arch=arch, norm=norm, residual=residual, dim_feat=dim_feat, depth=depth, seed=seed, batch_size=batch_size, learning_rate=(initial_lr, lr_interval, lr_scale))

    log(log_path, args_str)

    optimizer = optim.Adam(task.model.parameters(), lr=initial_lr)
    if lr_interval > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_interval, gamma=lr_scale, verbose=True)
    elif lr_interval < 0:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=lr_scale, patience=-lr_interval, verbose=True)
    else:
        raise ValueError("lr_interval cannot be 0")

    task.train(epochs, optimizer, scheduler, save=(log_path, save_dir, save_name) if save else None)


def run_test(task: Task, model_paths: List[str]):
    for model_path in model_paths:
        if not Path(model_path).exists():
            continue

        match = MODEL_REGEX.search(model_path)
        assert match is not None
        arch = match.group("arch")
        norm = match.group("norm")
        dim_feat = int(match.group("dim"))
        depth = int(match.group("depth"))

        task.build_model(arch, norm, False, dim_feat, depth)
        task.model.load_state_dict(th.load(model_path, map_location=DEVICE))
        valid_perf = task.evaluate(task.valid_loader, silent=True)
        test_perf = task.evaluate(task.test_loader, silent=True)

        print(f"{model_path}: val={valid_perf} test={test_perf}")


def main(
        args_str: str,
        dataset_name: str,
        batch_size: int,
        arch: str,
        norm: str,
        residual: bool,
        dim_feat: int,
        depth: int,
        seed: int,
        epochs: int,
        initial_lr: float,
        lr_interval: int,
        lr_scale: float,
        *,
        train: bool=False,
        test: Optional[List[str]]=None,
        save: bool=True,
        code2_use_subtoken: bool=False,
        code2_decoder_type: Optional[str]=None,
):

    if dataset_name == "ogbg-code2":
        task = OGBCodeTask(dataset_name, DEVICE, save, code2_use_subtoken, code2_decoder_type)
    else:
        raise NotImplementedError
    task.load_dataset(batch_size)

    if train:
        run_training(args_str, task, arch, norm, residual, dim_feat, depth, seed, epochs, batch_size, initial_lr, lr_interval, lr_scale, save=save)
    elif test:
        run_test(task, test)
