from typing import Callable

from dgl import function as fn
from dgl.nn.pytorch import GlobalAttentionPooling
import torch as th
from torch import nn
from torch.nn import functional as F


class GINLayer(nn.Module):
    def __init__(self, dim_feat: int, edge_encoder: nn.Module, edge_feat: str="feat"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_feat, 2 * dim_feat),
            nn.ReLU(),
            nn.Linear(2 * dim_feat, dim_feat),
        )
        self.eps = nn.Parameter(th.Tensor([0]))
        self.edge_encoder = edge_encoder
        self._dim_feat = dim_feat
        self._edge_feat = edge_feat

    def forward(self, graph, feat):
        graph.ndata["h"] = feat
        graph.apply_edges(lambda edges: {"e": F.relu(edges.src["h"] + self.edge_encoder(edges.data[self._edge_feat]).view((-1, self._dim_feat)))})
        graph.update_all(fn.copy_e("e", "m"), fn.sum("m", "a"))
        feat = self.mlp((1 + self.eps) * graph.ndata.pop("h") + graph.ndata.pop("a"))
        return feat


class GraphNorm(nn.Module):
    def __init__(self, dim_feat: int):
        super().__init__()

        self.weight = nn.Parameter(th.ones(dim_feat))
        self.bias = nn.Parameter(th.zeros(dim_feat))
        self.mean_scale = nn.Parameter(th.ones(dim_feat))

    def forward(self, graph, feat):
        batch_list = graph.batch_num_nodes().to(device=graph.device, dtype=th.int64)
        batch_index = th.arange(graph.batch_size, device=graph.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (feat.dim() - 1)).expand_as(feat)

        mean = th.zeros(graph.batch_size, *feat.shape[1:], device=graph.device)
        mean = mean.scatter_add_(0, batch_index, feat)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = feat - mean * self.mean_scale

        std = th.zeros(graph.batch_size, *feat.shape[1:], device=graph.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)

        return self.weight * sub / std + self.bias


class GNNBase(nn.Module):
    def __init__(
            self,
            layer: str,
            norm: str,
            res: bool,
            dim_feat: int,
            depth: int,
            edge_encoder: Callable[[], nn.Module],
            *,
            dropout: bool=True,
            edge_feat: str="feat"
    ):
        super().__init__()

        self._dim_feat = dim_feat
        self._depth = depth
        self._res = res
        self._dropout = dropout

        if layer == "gin":
            self.layers = nn.ModuleList([GINLayer(dim_feat, edge_encoder(), edge_feat) for _ in range(depth)])
        else:
            raise ValueError(f"invalid layer type: {layer}")

        if norm == "none":
            self.norms = None
        elif norm == "graphnorm":
            self.norms = nn.ModuleList([GraphNorm(dim_feat) for _ in range(depth)])
        else:
            raise ValueError(f"invalid norm: {norm}.")


class GNNSimple(GNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pooling = GlobalAttentionPooling(nn.Sequential(
            nn.Linear(self._dim_feat, 2 * self._dim_feat),
            nn.ReLU(),
            nn.Linear(2 * self._dim_feat, 1),
        ))

    def forward(self, graph, feat):
        for d in range(self._depth):
            feat_in = feat
            feat = self.layers[d](graph, feat)
            if self.norms:
                feat = self.norms[d](graph, feat)
            if d < self._depth - 1:
                feat = F.relu(feat)
            if self._dropout:
                feat = F.dropout(feat, training=self.training)
            if self._res:
                feat = feat + feat_in

        return self.pooling(graph, feat)

    def get_emb(self, graph, feat):
        return self.forward(graph, feat).unsqueeze(0)
