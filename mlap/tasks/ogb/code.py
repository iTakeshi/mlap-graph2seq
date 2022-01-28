from collections import defaultdict
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import dgl
import numpy as np
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
import pandas as pd
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlap.nn.decoders import LinearDecoder, LSTMDecoder
from mlap.tasks.base import Task, TeacherForcing
from mlap.util import encode_seq_to_arr, decode_arr_to_seq


class OGBCodeTask(Task):
    def __init__(
            self,
            dataset_name: str,
            device: th.device,
            save: bool,
            use_subtoken: bool,
            decoder_type: str
    ) -> None:

        super().__init__(dataset_name, device, save)
        self._use_subtoken = use_subtoken
        self._decoder_type = decoder_type

    def load_dataset(self, batch_size: int) -> None:
        self.dataset = DglGraphPropPredDataset(name=self.dataset_name)
        if "feat" in self.dataset[0][0].ndata:
            self.dataset.dim_node = self.dataset[0][0].ndata["feat"].shape[1]
        else:
            self.dataset.dim_node = 0
        if "feat" in self.dataset[0][0].edata:
            self.dataset.dim_edge = self.dataset[0][0].edata["feat"].shape[1]
        else:
            self.dataset.dim_edge = 0

        self._max_depth = 20
        self._max_seq_len = 5
        self._num_vocab = 5000
        self._num_nodetypes = len(pd.read_csv(Path(self.dataset.root) / "mapping" / "typeidx2type.csv.gz")["type"])

        split_idx = self.dataset.get_idx_split()
        self._vocab2idx, self._id2vocab = _get_vocab_mapping([l for _, l in self.dataset[split_idx["train"]]], self._num_vocab)

        if self._use_subtoken:
            pickle_path = Path(self.dataset.root) / "saved" / "loaders-subtoken.pkl"
        else:
            pickle_path = Path(self.dataset.root) / "saved" / "loaders.pkl"

        if pickle_path.exists():
            with open(pickle_path, "rb") as f:
                train_samples = pickle.load(f)
                valid_samples = pickle.load(f)
                test_samples = pickle.load(f)
                self._num_nodeattrs = pickle.load(f)

        else:
            if self._use_subtoken:
                def _subtokenize_attr(attr: str) -> List[str]:
                    def camel_case_split(s: str) -> List[str]:
                        matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", s)
                        return [m.group(0).lower() for m in matches]

                    if attr in ["__NONE__", "__UNK__"]:
                        res = [attr]
                    else:
                        res = []
                        for part in str(attr).split('_'):
                            res.extend(camel_case_split(part))

                    if len(res) >= self._max_seq_len:
                        return res[:self._max_seq_len]
                    else:
                        return res + ["__NONE__"] * (self._max_seq_len - len(res))

                attr_subtokens = {}
                for i, _, a in pd.read_csv(Path(self.dataset.root) / "mapping" / "attridx2attr.csv.gz").itertuples():
                    attr_subtokens[i] = _subtokenize_attr(a)

                def get_split(idx, is_training):
                    samples = []
                    for i in tqdm(idx):
                        samples.append((_augment_edge(_subtokenize(self.dataset[i][0], attr_subtokens, used_subtokens, is_training)), self.dataset[i][1]))
                    return samples

                used_subtokens = {"__NONE__": 0, "__UNK__": 1, "__NUM__": 2}
                train_samples = get_split(split_idx["train"], True)
                self._num_nodeattrs = len(used_subtokens)
                valid_samples = get_split(split_idx["valid"], False)
                test_samples = get_split(split_idx["test"], False)

            else:
                self._num_nodeattrs = len(pd.read_csv(Path(self.dataset.root) / "mapping" / "attridx2attr.csv.gz")["attr"])

                def get_split(idx):
                    samples = []
                    for i in tqdm(idx):
                        samples.append((_augment_edge(self.dataset[i][0]), self.dataset[i][1]))
                    return samples

                train_samples = get_split(split_idx["train"])
                valid_samples = get_split(split_idx["valid"])
                test_samples = get_split(split_idx["test"])

            pickle_path.parent.mkdir(exist_ok=True)
            with open(pickle_path, "wb") as f:
                pickle.dump(train_samples, f)
                pickle.dump(valid_samples, f)
                pickle.dump(test_samples, f)
                pickle.dump(self._num_nodeattrs, f)

        def collate(samples) -> Tuple[dgl.DGLGraph, List[List[str]]]:
            graphs, labels = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            return batched_graph, labels

        self.train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True, collate_fn=collate)
        self.valid_loader = DataLoader(valid_samples, batch_size=batch_size, shuffle=False, collate_fn=collate)
        self.test_loader = DataLoader(test_samples, batch_size=batch_size, shuffle=False, collate_fn=collate)

    def _build_model(self, arch: str, norm: str, residual: bool, dim_feat: int, depth: int) -> nn.Module:
        edge_encoder = lambda: nn.Embedding(4, dim_feat)
        gnn = self._build_gnn(arch, norm, residual, dim_feat, depth, edge_encoder)
        return CodeEncDec(
            gnn,
            dim_feat,
            self._max_depth,
            self._max_seq_len,
            self._num_nodetypes,
            self._num_nodeattrs,
            self._vocab2idx,
            self._decoder_type,
        ).to(self.device)

    def _evaluate_score(self, y_true: List[Any], y_pred: List[th.Tensor], batch_sizes: List[int]) -> float:
        y_true = sum(y_true, [])
        y_pred = th.vstack([th.hstack([t.argmax(dim=1).view(b, -1) for t in l]) for l, b in zip(y_pred, batch_sizes)])
        y_pred = [decode_arr_to_seq(a, self._id2vocab, self._vocab2idx) for a in y_pred]
        metric: str = self.dataset.eval_metric
        return Evaluator(self.dataset_name).eval({"seq_ref": y_true, "seq_pred": y_pred})[metric]

    def _loss(self, out: th.Tensor, labels: List[List[str]]) -> th.Tensor:
        batched_label = th.vstack([encode_seq_to_arr(label, self._vocab2idx, self._max_seq_len) for label in labels])
        return sum([nn.CrossEntropyLoss()(out[i], batched_label[:, i].to(device=self.device)) for i in range(self._max_seq_len)])

    def get_emb(self, save_emb_name: str):
        for name, loader in {"train": self.train_loader, "valid": self.valid_loader, "test": self.test_loader}.items():
            embs = []
            labels = []
            for batch in tqdm(loader):
                g, l = batch
                with th.no_grad():
                    embs.append(self.model.get_emb(g.to(self.device)).detach().cpu())
                    labels.extend(l)
            embs = th.cat(embs, dim=1)
            labels = th.vstack([encode_seq_to_arr(label, self._vocab2idx, self._max_seq_len) for label in labels])

            save_path = self.emb_dir / f"{save_emb_name}_{name}"
            np.savez(save_path, embs=embs.numpy(), labels=labels.numpy())


class CodeEncDec(TeacherForcing, nn.Module):
    def __init__(
            self,
            gnn: nn.Module,
            dim_feat: int,
            max_depth: int,
            max_seq_len: int,
            num_nodetypes: int,
            num_nodeattrs: int,
            vocab2idx: Dict[str, int],
            decoder_type: str,
    ):

        super().__init__()
        self._dim_feat = dim_feat
        self._max_depth = max_depth
        self._max_seq_len = max_seq_len
        self._vocab2idx = vocab2idx
        self._decoder_type = decoder_type

        self.type_encoder = nn.Embedding(num_nodetypes, dim_feat)
        self.attr_encoder = nn.Embedding(num_nodeattrs, dim_feat)
        self.depth_encoder = nn.Embedding(max_depth + 1, dim_feat)
        self.node_mlp = nn.Sequential(
            nn.Linear(3 * dim_feat, 2 * dim_feat),
            nn.ReLU(),
            nn.Linear(2 * dim_feat, dim_feat),
        )
        self.gnn = gnn

        if decoder_type == "linear":
            self.linear_decoder = LinearDecoder(dim_feat, max_seq_len, vocab2idx)
        elif decoder_type == "lstm":
            self.lstm_decoder = LSTMDecoder(dim_feat, max_seq_len, vocab2idx)

    def forward(self, graph: dgl.DGLGraph, labels: Any) -> List[th.Tensor]:
        feats = self.get_emb(graph)  # (L+1, batch_size, dim_feat)

        if self._decoder_type == "linear":
            return self.linear_decoder(graph, feats, labels)
        elif self._decoder_type == "lstm":
            return self.lstm_decoder(graph, feats, labels)

    def get_emb(self, graph: dgl.DGLGraph) -> th.Tensor:
        type_emb = self.type_encoder(graph.ndata["feat"][:, 0])
        attr_emb = (self.attr_encoder(graph.ndata["feat"][:, 1:]) * (graph.ndata["feat"][:, 1:] > 0).unsqueeze(-1)).sum(dim=1)
        depth = graph.ndata["depth"].view(-1)
        depth[depth > self._max_depth] = self._max_depth
        depth_emb = self.depth_encoder(depth)
        feat = self.node_mlp(th.hstack((type_emb, attr_emb, depth_emb)))
        return self.gnn.get_emb(graph, feat)


def _get_vocab_mapping(words_list: List[List[str]], num_vocab: int) -> Tuple[Dict[str, int], List[str]]:
    vocab_count = defaultdict(int)
    for words in tqdm(words_list):
        for word in words:
            vocab_count[word] += 1
    idx2vocab = ["__SOS__", "__UNK__", "__EOS__"]
    idx2vocab += list(list(zip(*sorted([(c, w) for w, c in vocab_count.items()], reverse=True)[:num_vocab]))[1])
    vocab2idx = {w: i for i, w in enumerate(idx2vocab)}

    # test
    for idx, vocab in enumerate(idx2vocab):
        assert(idx == vocab2idx[vocab])
    assert(vocab2idx["__SOS__"] == 0)
    assert(vocab2idx["__UNK__"] == 1)
    assert(vocab2idx["__EOS__"] == 2)

    return vocab2idx, idx2vocab


def _augment_edge(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    num_ast_edges = graph.num_edges()
    src_ast = th.hstack((graph.edges()[0], graph.edges()[1]))
    dst_ast = th.hstack((graph.edges()[1], graph.edges()[0]))
    attr_ast = th.vstack((th.zeros((num_ast_edges, 1)), th.ones((num_ast_edges, 1))))

    terminals = th.where(graph.ndata["is_attributed"] == 1)[0]
    num_nt_edges = terminals.shape[0] - 1
    src_nt = th.hstack((terminals[:-1], terminals[1:]))
    dst_nt = th.hstack((terminals[1:], terminals[:-1]))
    attr_nt = th.vstack((th.ones((num_nt_edges, 1)) * 2, th.ones((num_nt_edges, 1)) * 3))

    graph.remove_edges(np.arange(num_ast_edges))
    graph.add_edges(th.hstack((src_ast, src_nt)), th.hstack((dst_ast, dst_nt)), {"feat": th.vstack((attr_ast, attr_nt)).to(th.int64)})

    return graph


def _subtokenize(
        graph: dgl.DGLGraph,
        attr_subtokens: Dict[int, List[str]],
        used_subtokens: Dict[str, int],
        is_training: bool,
) -> dgl.DGLGraph:

    feat = th.hstack((graph.ndata["feat"][:, 0].view(-1, 1), th.zeros((graph.ndata["feat"].shape[0], 5)).to(graph.ndata["feat"])))

    for i in range(graph.ndata["feat"].shape[0]):
        if int(graph.ndata["feat"][i, 0]) == 67:  # Num
            feat[i, 1] = used_subtokens["__NUM__"]
        elif int(graph.ndata["feat"][i, 1]) == 10028:  # __NONE__
            pass
        else:
            for j, s in enumerate(attr_subtokens[int(graph.ndata["feat"][i, 1])]):
                if is_training:
                    if s not in used_subtokens:
                        used_subtokens[s] = len(used_subtokens)
                    feat[i, j + 1] = used_subtokens[s]
                else:
                    if s in used_subtokens:
                        feat[i, j + 1] = used_subtokens[s]
                    else:
                        feat[i, j + 1] = used_subtokens["__UNK__"]

    graph.ndata["feat"] = feat

    return graph
