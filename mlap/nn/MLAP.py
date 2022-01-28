from dgl.nn.pytorch import GlobalAttentionPooling
import torch as th
from torch import nn
from torch.nn import functional as F

from mlap.nn.base import GNNBase


class MLAPBase(GNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.poolings = nn.ModuleList(
            [GlobalAttentionPooling(nn.Sequential(
                nn.Linear(self._dim_feat, 2 * self._dim_feat),
                nn.ReLU(),
                nn.Linear(2 * self._dim_feat, 1),
            )) for _ in range(self._depth)]
        )

    def forward(self, graph, feat):
        self._graph_embs = []

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

            self._graph_embs.append(self.poolings[d](graph, feat))

        return self._aggregate()

    def _aggregate(self):
        raise NotImplementedError

    def get_emb(self, graph, feat):
        out = self.forward(graph, feat)
        self._graph_embs.append(out)
        return th.stack(self._graph_embs, dim=0)


class MLAPSum(MLAPBase):
    def _aggregate(self):
        return th.stack(self._graph_embs, dim=0).sum(dim=0)


class MLAPWeighted(MLAPBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(th.ones(self._depth, 1, 1))

    def _aggregate(self):
        a = F.softmax(self.weight, dim=0)
        h = th.stack(self._graph_embs, dim=0)
        return (a * h).sum(dim=0)
