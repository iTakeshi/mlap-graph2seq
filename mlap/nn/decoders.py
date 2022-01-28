from typing import Dict, List

import dgl
import torch as th
from torch import nn
from torch.nn import functional as F

from mlap.util import encode_seq_to_arr


class LinearDecoder(nn.Module):
    def __init__(self, dim_feat: int, max_seq_len: int, vocab2idx: Dict[str, int]):
        super().__init__()
        self._max_seq_len = max_seq_len
        self._vocab2idx = vocab2idx

        self.decoders = nn.ModuleList([nn.Linear(dim_feat, len(vocab2idx)) for _ in range(max_seq_len)])

    def forward(self, graph: dgl.DGLGraph, feats: th.Tensor, labels: List[List[str]]) -> List[th.Tensor]:
        return [d(feats[-1]) for d in self.decoders]


class LSTMDecoder(nn.Module):
    def __init__(self, dim_feat: int, max_seq_len: int, vocab2idx: Dict[str, int]):
        super().__init__()
        self._max_seq_len = max_seq_len
        self._vocab2idx = vocab2idx

        self.lstm = nn.LSTMCell(dim_feat, dim_feat)
        self.w_hc = nn.Linear(dim_feat * 2, dim_feat)
        self.layernorm = nn.LayerNorm(dim_feat)
        self.vocab_encoder = nn.Embedding(len(vocab2idx), dim_feat)
        self.vocab_bias = nn.Parameter(th.zeros(len(vocab2idx)))

    def forward(self, graph: dgl.DGLGraph, feats: th.Tensor, labels: List[List[str]]) -> List[th.Tensor]:
        if self.training:
            # teacher forcing
            batched_label = th.vstack([encode_seq_to_arr(label, self._vocab2idx, self._max_seq_len - 1) for label in labels])
            batched_label = th.hstack((th.zeros((graph.batch_size, 1), dtype=th.int64), batched_label))
            true_emb = self.vocab_encoder(batched_label.to(device=graph.device))

        h_t, c_t = feats[-1].clone(), feats[-1].clone()
        feats = feats.transpose(0, 1)  # (batch_size, L + 1, dim_feat)
        out = []
        pred_emb = self.vocab_encoder(th.zeros((graph.batch_size), dtype=th.int64, device=graph.device))

        vocab_mat = self.vocab_encoder(th.arange(len(self._vocab2idx), dtype=th.int64, device=graph.device))

        for i in range(self._max_seq_len):
            if self.training:
                _in = true_emb[:, i]
            else:
                _in = pred_emb
            h_t, c_t = self.lstm(_in, (h_t, c_t))

            a = F.softmax(th.bmm(feats, h_t.unsqueeze(-1)).squeeze(-1), dim=1)  # (batch_size, L + 1)
            context = th.bmm(a.unsqueeze(1), feats).squeeze(1)
            pred_emb = th.tanh(self.layernorm(self.w_hc(th.hstack((h_t, context)))))  # (batch_size, dim_feat)

            out.append(th.matmul(pred_emb, vocab_mat.T) + self.vocab_bias.unsqueeze(0))

        return out
