import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import random
import numpy as np
import torch as th
from torch import cuda


def log(path: Optional[Path], msg: str):
    print(msg)

    if path:
        with open(path, "a") as f:
            f.write(msg)
            f.write("\n")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if cuda.is_available():
        cuda.manual_seed(seed)


def get_repo_root() -> Path:
    env_mlap_root = os.environ.get("MLAP_ROOT")
    return Path(env_mlap_root) if env_mlap_root else Path(__file__).parent.parent


def encode_seq_to_arr(seq: List[str], vocab2idx: Dict[str, int], max_seq_len: int) -> th.Tensor:
    seq = seq[:max_seq_len] + ["__EOS__"] * max(0, max_seq_len - len(seq))
    return th.tensor([vocab2idx[w] if w in vocab2idx else vocab2idx["__UNK__"] for w in seq], dtype=th.int64)


def decode_arr_to_seq(arr: Union[List[int], th.Tensor], idx2vocab: List[str], vocab2idx: Dict[str, int]) -> List[str]:
    if isinstance(arr, th.Tensor):
        arr = arr.tolist()
    if vocab2idx["__EOS__"] in arr:
        arr = arr[:arr.index(vocab2idx["__EOS__"])]
    return [idx2vocab[i] for i in arr]
