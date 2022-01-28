from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import dgl
import numpy as np
import torch as th
from torch import nn, optim
from tqdm import tqdm

from mlap.nn.base import GNNSimple
from mlap.nn.MLAP import MLAPSum, MLAPWeighted
from mlap.util import log, set_seed, get_repo_root


GNN_CLASS = {
    "simple": GNNSimple,

    "mlap-sum": MLAPSum,
    "mlap-weighted": MLAPWeighted,
}


class TeacherForcing(nn.Module):
    def forward(self, graph: dgl.DGLGraph, labels: Any) -> th.Tensor:
        raise NotImplementedError


class Task:
    def __init__(self, dataset_name: str, device: th.device, save: bool) -> None:
        self.dataset_name = dataset_name
        self.device = device
        self.save = save

        if not self.log_dir.exists():
            self.log_dir.mkdir()
        if not self.model_dir.exists():
            self.model_dir.mkdir()

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    @property
    def log_dir(self) -> Path:
        return get_repo_root() / "log" / self.dataset_name

    @property
    def model_dir(self) -> Path:
        return get_repo_root() / "model" / self.dataset_name

    def load_dataset(self, batch_size: int) -> None:
        raise NotImplementedError

    def build_model(self, arch: str, norm: str, res: bool, dim_feat: int, depth: int) -> None:
        self.model = self._build_model(arch, norm, res, dim_feat, depth)

    def _build_model(self, arch: str, norm: str, res: bool, dim_feat: int, depth: int) -> nn.Module:
        raise NotImplementedError

    def _build_gnn(self, arch: str, *args, **kwargs) -> nn.Module:
        i = arch.index("-")
        self._current_gnn = GNN_CLASS[arch[(i + 1):]](arch[:i], *args, **kwargs)
        return self._current_gnn

    def set_seed(self, seed: int):
        self._seed = seed
        set_seed(seed)

    def evaluate(self, loader, *, silent=False) -> float:
        self.model.eval()
        batch_sizes = []
        y_true = []
        y_pred = []

        for batch in tqdm(loader, disable=silent):
            g, labels = batch
            with th.no_grad():
                if isinstance(self.model, TeacherForcing):
                    out = [t.detach().cpu() for t in self.model(g.to(self.device), None)]
                else:
                    out = self.model(g.to(self.device)).detach().cpu()
            batch_sizes.append(g.batch_size)
            y_true.append(labels)
            y_pred.append(out)

        return self._evaluate_score(y_true, y_pred, batch_sizes)

    def _evaluate_score(self, y_true: List[Any], y_pred: List[th.Tensor], batch_sizes: List[int]) -> float:
        raise NotImplementedError

    def train(
            self,
            epochs: int,
            optimizer: optim.Optimizer,
            scheduler: Optional[Union[optim.lr_scheduler.StepLR, optim.lr_scheduler.ReduceLROnPlateau]],
            *,
            save: Optional[Tuple[Path, Path, str]]=None,
    ) -> Tuple[List[float], List[float]]:

        train_curve = []
        valid_curve = []

        log_path = save[0] if save else None

        for epoch in range(1, epochs + 1):
            print("Training...")
            self.model.train()
            with tqdm(self.train_loader) as pbar:
                for i, batch in enumerate(pbar):
                    g, labels = batch
                    if isinstance(self.model, TeacherForcing):
                        out = self.model(g.to(self.device), labels)
                    else:
                        out = self.model(g.to(self.device))
                    loss = self._loss(out, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pbar.set_description(f"Epoch {epoch: >3}, batch {i: > 5} loss: {loss.data:.3f}")

            print("Evaluating...")
            # train_perf = self.evaluate(self.train_loader)
            train_perf = 0.0
            valid_perf = self.evaluate(self.valid_loader)
            log(log_path, f"Epoch {epoch}, train {train_perf}, valid {valid_perf}")
            train_curve.append(train_perf)
            valid_curve.append(valid_perf)

            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(valid_perf)
                else:
                    scheduler.step()

            if save:
                _, save_dir, save_name = save
                th.save(self.model.state_dict(), save_dir / f"{save_name}_e{epoch}")

        log(log_path, f"Best validation score: {np.max(valid_curve)}")
        return train_curve, valid_curve

    def _loss(self, out: th.Tensor, labels: th.Tensor) -> th.Tensor:
        raise NotImplementedError
