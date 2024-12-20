from typing import Tuple
from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor
from torch_scatter import scatter_max

from sklearn.model_selection import train_test_split
import yaml


def select_latest_nids(nids: Tensor, ts: Tensor) -> Tuple[Tensor, Tensor]:
    # Using unique to re-index node ids
    unique_ids, unique_index = torch.unique(nids, return_inverse=True)
    # find indices of the last events of nodes
    # unique_idx starts from 0
    _, max_index = scatter_max(ts, unique_index)
    return unique_ids, max_index


def anonymized_reindex(hist_nids: np.ndarray) -> np.ndarray:
    mask = hist_nids == 0
    out_nids = np.zeros_like(hist_nids)
    for i, line in enumerate(hist_nids):
        od = OrderedDict.fromkeys(line[::-1])  # get unique values with preserved order
        reindex_map = {k: j+1 for j, k in enumerate(od.keys())}
        out_nids[i] = [reindex_map[j] for j in line]
    out_nids[mask] = 0
    return out_nids


def set_anonymized_encoding(walk_nids: np.ndarray) -> np.ndarray:
    """
    walk_nids: [bs, n_walks, walk_length]
    """
    bs, n, length = walk_nids.shape
    batch_codes = []
    batch_dicts = []
    for i in range(bs):
        walks = walk_nids[i]  # [n_walks, walk_length]
        unique_ids, inv_idx = np.unique(walks.flatten(), return_inverse=True)
        counts = (walks[:, :, None] == unique_ids).sum(0).T  # [n_unique, length]
        codes = counts[inv_idx].reshape(n, length, length)
        batch_codes.append(codes)
        batch_dicts.append({k: v for k, v in zip(unique_ids, counts)})
    batch_codes = np.stack(batch_codes, 0)
    return batch_codes, batch_dicts

class ContextTargetSpliter():
    def __init__(self, context_split):
        self.context_split = context_split

    def __call__(self, data):
        c_src, t_src, c_pos_dst, t_pos_dst, c_neg_dst, t_neg_dst, c_ts_src, t_ts_src, c_ts_pos_dst, t_ts_pos_dst, c_ts_neg_dst, t_ts_neg_dst = train_test_split(
            *data,
            train_size=self.context_split)
        return [c_src, c_pos_dst, c_neg_dst], [t_src, t_pos_dst, t_neg_dst], [c_ts_src, c_ts_pos_dst, c_ts_neg_dst], [
            t_ts_src, t_ts_pos_dst, t_ts_neg_dst]


def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = []
    memory_param = []
    gnn_param = conf['gnn'][0]
    np_param = conf['np'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, np_param, train_param