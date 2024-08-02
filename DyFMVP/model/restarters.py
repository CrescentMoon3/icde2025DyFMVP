from typing import Optional, Tuple, Union
import warnings

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.data_loader import ComputationGraph
from ..data.graph import Graph
from .basic_modules import MergeLayer
from .feature_getter import FeatureGetter
from .time_encoding import TimeEncode
from .model_utils import anonymized_reindex, set_anonymized_encoding

from torchdiffeq import odeint_adjoint
from .layers import TimeEncode as TimeEncode_node
from .temporal_agg_modules import GraphAttnEmbedding

class ODESolver_node(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(self, r_dim, h_dim, t_dim, time_enc, r_tol=1e-6, a_tol=1e-7, method="dopri5"):
        super(ODESolver_node, self).__init__()
        self.a_tol = a_tol
        self.r_tol = r_tol
        self.method = method
        self.ode_func = ODEFunc_node(r_dim, h_dim, t_dim,
                                time_enc, self.start_time, self.end_time)

    def forward(self, r, start_t, end_t):

        initial_state = (r, end_t, start_t)
        tt = torch.tensor([self.start_time, self.end_time]).to(r)  # [0, 1]
        solution = odeint_adjoint(
            self.ode_func,
            initial_state,
            tt,
            rtol=self.r_tol,
            atol=self.a_tol,
            method=self.method,
        )
        r_final, _, _ = solution
        r_final = r_final[-1, :]
        return r_final


class ODEFunc_node(nn.Module):
    def __init__(self, z_dim, h_dim, t_dim, time_enc, start_time, end_time):
        super(ODEFunc_node, self).__init__()
        self.start_time = start_time
        self.end_time = end_time
        # Timestamp's dimension =1
        self.time_enc = time_enc
        ode_layers = [
            nn.Linear(z_dim, h_dim),
            nn.Tanh()
        ]
        self.latent_odefunc = nn.Sequential(*ode_layers)

    def forward(self, s, x):

        z, ts, t_0 = x
        ratio = (ts - t_0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t_0
        with torch.enable_grad():
            z = z.requires_grad_(True)
            t = t.requires_grad_(True)
            t = self.time_enc(t)
            dz = self.latent_odefunc(z + t)

            dz = dz * ratio
        return dz, ts, t_0


class MergeAUG_node(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        # self.relu = nn.ReLU()

    def forward(self, x1, x2):
        h = torch.cat([x1, x2], dim=1)
        h = self.norm(h)
        h = self.dropout(h)

        h = self.fc(h)
        
        return h 


class Restarter(nn.Module):
    def __init__(self, raw_feat_getter: FeatureGetter, graph: Graph):
        super().__init__()
        self.raw_feat_getter = raw_feat_getter
        self.graph = graph

        self.n_nodes = self.raw_feat_getter.n_nodes
        self.nfeat_dim = self.raw_feat_getter.nfeat_dim
        self.efeat_dim = self.raw_feat_getter.efeat_dim

        self.time_encoder = TimeEncode(dim=self.nfeat_dim)
        self.tfeat_dim = self.time_encoder.dim

    def forward(self, nids: Tensor, ts: Tensor,
                computation_graph: Optional[ComputationGraph]=None
               ) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError


class SeqRestarter_naligned(Restarter):
    def __init__(self, raw_feat_getter: FeatureGetter, graph: Graph,
                 *, hist_len: int=20, n_head=2, dropout=0.1):
        super().__init__(raw_feat_getter, graph)

        self.hist_len = hist_len

        self.anony_emb = nn.Embedding(self.hist_len + 1, self.nfeat_dim)

        self.d_model = self.nfeat_dim * 3 + self.efeat_dim + self.tfeat_dim
        self.mha_fn = nn.MultiheadAttention(self.d_model, n_head, dropout)
        self.out_fn = nn.Linear(self.d_model, self.nfeat_dim)
        self.merger = MergeLayer(self.nfeat_dim, self.d_model - self.tfeat_dim,
                                 self.nfeat_dim, self.nfeat_dim, dropout=dropout)

    def forward(self, nids: Tensor, ts: Tensor,
                computation_graph: Optional[ComputationGraph]=None
               ) -> Tuple[Tensor, Tensor, Tensor]:

        if computation_graph is None:
            device = nids.device
            hist_nids, hist_eids, hist_ts, hist_dirs = self.graph.get_history(
                nids.cpu().numpy(), ts.cpu().numpy(), self.hist_len)
            anonymized_ids = anonymized_reindex(hist_nids)
            hist_nids = torch.from_numpy(hist_nids).to(device).long()  # [bs, len]
            anonymized_ids = torch.from_numpy(anonymized_ids).to(device).long()
            hist_eids = torch.from_numpy(hist_eids).to(device).long()
            hist_ts = torch.from_numpy(hist_ts).to(device).float()
            hist_dirs = torch.from_numpy(hist_dirs).to(device).long()
        else:
            data = computation_graph.restart_data
            hist_nids = data.hist_nids
            anonymized_ids = data.anonymized_ids
            hist_eids = data.hist_eids
            hist_ts = data.hist_ts
            hist_dirs = data.hist_dirs

        bs, hist_len = hist_nids.shape
        mask = (hist_nids == 0)
        mask[:, -1] = False
        invalid_rows = mask.all(1, keepdims=True)

        r_nids = nids.unsqueeze(1).repeat(1, hist_len)
        src_nids = r_nids * hist_dirs + hist_nids * (1-hist_dirs)
        dst_nids = r_nids * (1-hist_dirs) + hist_nids * hist_dirs

        src_vals = self.raw_feat_getter.get_node_embeddings(src_nids)
        dst_vals = self.raw_feat_getter.get_node_embeddings(dst_nids)
        edge_vals = self.raw_feat_getter.get_edge_embeddings(hist_eids)
        anony_vals = self.anony_emb(anonymized_ids)
        ts_vals = self.time_encoder(hist_ts[:, -1].unsqueeze(1) - hist_ts)
        full_vals = torch.cat([src_vals, dst_vals, anony_vals, edge_vals, ts_vals], 2)

        last_event_feat = full_vals[:, -1, :self.d_model - self.tfeat_dim]
        full_vals[:, -1, :self.d_model - self.tfeat_dim] = 0.
        qkv = full_vals.transpose(0, 1)
        out, _ = self.mha_fn(qkv, qkv, qkv, key_padding_mask=mask)
        h_prev_left = self.out_fn(F.relu(out.mean(0)))
        h_prev_right = self.merger(h_prev_left, last_event_feat)
        h_prev_left = h_prev_left.masked_fill(invalid_rows, 0.)
        h_prev_right = h_prev_right.masked_fill(invalid_rows, 0.)
        prev_ts = hist_ts[:, -1]
        return h_prev_left, h_prev_right, prev_ts


class SeqRestarter(Restarter):
    def __init__(self, raw_feat_getter: FeatureGetter, graph: Graph,
                 *, hist_len: int=20, n_head=2, dropout=0.1):
        super().__init__(raw_feat_getter, graph)

        self.hist_len = hist_len

        self.anony_emb = nn.Embedding(self.hist_len + 1, self.nfeat_dim)

        self.d_model = self.nfeat_dim * 3 + self.efeat_dim + self.tfeat_dim
        self.mha_fn = nn.MultiheadAttention(self.d_model, n_head, dropout)
        self.out_fn = nn.Linear(self.d_model, self.nfeat_dim)
        self.merger = MergeLayer(self.nfeat_dim, self.d_model - self.tfeat_dim, 
                                 self.nfeat_dim, self.nfeat_dim, dropout=dropout)
    

    def forward(self, nids: Tensor, ts: Tensor,
                computation_graph: Optional[ComputationGraph]=None
               ) -> Tuple[Tensor, Tensor, Tensor]:

        if computation_graph is None:
            device = nids.device
            hist_nids, hist_eids, hist_ts, hist_dirs = self.graph.get_history(
                nids.cpu().numpy(), ts.cpu().numpy(), self.hist_len)
            anonymized_ids = anonymized_reindex(hist_nids)
            hist_nids = torch.from_numpy(hist_nids).to(device).long()  # [bs, len]
            anonymized_ids = torch.from_numpy(anonymized_ids).to(device).long()
            hist_eids = torch.from_numpy(hist_eids).to(device).long()
            hist_ts = torch.from_numpy(hist_ts).to(device).float()
            hist_dirs = torch.from_numpy(hist_dirs).to(device).long()
        else:
            data = computation_graph.restart_data
            hist_nids = data.hist_nids
            anonymized_ids = data.anonymized_ids
            hist_eids = data.hist_eids
            hist_ts = data.hist_ts
            hist_dirs = data.hist_dirs

        bs, hist_len = hist_nids.shape
        mask = (hist_nids == 0)
        mask[:, -1] = False
        invalid_rows = mask.all(1, keepdims=True)

        r_nids = nids.unsqueeze(1).repeat(1, hist_len)
        src_nids = r_nids * hist_dirs + hist_nids * (1-hist_dirs)
        dst_nids = r_nids * (1-hist_dirs) + hist_nids * hist_dirs

        src_vals = self.raw_feat_getter.get_node_embeddings(src_nids)
        dst_vals = self.raw_feat_getter.get_node_embeddings(dst_nids)
        edge_vals = self.raw_feat_getter.get_edge_embeddings(hist_eids)
        anony_vals = self.anony_emb(anonymized_ids)
        ts_vals = self.time_encoder(hist_ts[:, -1].unsqueeze(1) - hist_ts)
        full_vals = torch.cat([src_vals, dst_vals, anony_vals, edge_vals, ts_vals], 2)

        last_event_feat = full_vals[:, -1, :self.d_model - self.tfeat_dim]
        full_vals[:, -1, :self.d_model - self.tfeat_dim] = 0.
        qkv = full_vals.transpose(0, 1)
        out, _ = self.mha_fn(qkv, qkv, qkv, key_padding_mask=mask)

        h_prev_left = self.out_fn(F.relu(out.mean(0)))

        h_prev_right = self.merger(h_prev_left, last_event_feat)
        h_prev_left = h_prev_left.masked_fill(invalid_rows, 0.)
        h_prev_right = h_prev_right.masked_fill(invalid_rows, 0.)
        prev_ts = hist_ts[:, -1]
        return h_prev_left, h_prev_right, prev_ts


class TaroRestarter(Restarter):
    def __init__(self, raw_feat_getter: FeatureGetter, graph: Graph,
                 *, hist_len: int=10, n_head=2, dropout=0.1):
        super().__init__(raw_feat_getter, graph)

        self.hist_len = hist_len

        self.anony_emb = nn.Embedding(self.hist_len + 1, self.nfeat_dim)

        self.d_model = self.nfeat_dim * 3 + self.efeat_dim + self.tfeat_dim
        self.mha_fn = nn.MultiheadAttention(self.d_model, n_head, dropout)
        self.out_fn = nn.Linear(self.d_model, self.nfeat_dim)
        self.merger = MergeLayer(self.nfeat_dim, self.d_model - self.tfeat_dim, 
                                 self.nfeat_dim, self.nfeat_dim, dropout=dropout)

        self.jump_update = nn.GRUCell(input_size=self.d_model - self.tfeat_dim, hidden_size=self.nfeat_dim)
        self.time_encoder_node = TimeEncode_node(self.nfeat_dim)
        self.ode_transform = ODESolver_node(self.nfeat_dim, self.nfeat_dim, self.nfeat_dim, self.time_encoder_node, r_tol=1e-5,
                                        a_tol=1e-7,
                                        method="dopri5")

        self.merge_aug_node = MergeAUG_node(self.nfeat_dim * 2, self.nfeat_dim)

        self.linear_transform = nn.Linear(self.nfeat_dim, self.nfeat_dim)


        self.n_neighbors = hist_len
        self.n_layers, self.n_head = 1, n_head
        self.temporal_embedding_fn = GraphAttnEmbedding(
            raw_feat_getter=self.raw_feat_getter,
            time_encoder=self.time_encoder, graph=graph,
            n_neighbors=self.n_neighbors, n_layers=self.n_layers, n_head=self.n_head,
            dropout=dropout
        )

    def forward(self, nids: Tensor, ts: Tensor,
                computation_graph: Optional[ComputationGraph]=None,
                flag: int=0, resize_ratio_st: float=1.0
               ) -> Tuple[Tensor, Tensor, Tensor]:

        if flag == 0:
            if computation_graph is None:
                device = nids.device
                hist_nids, hist_eids, hist_ts, hist_dirs = self.graph.get_history(
                    nids.cpu().numpy(), ts.cpu().numpy(), self.hist_len
                )
                hist_ts = torch.from_numpy(hist_ts).to(device).float()
            else:
                data = computation_graph.restart_data
                hist_ts = data.hist_ts
            prev_ts = hist_ts[:, -1]
            return prev_ts

        if computation_graph is None:
            device = nids.device
            hist_nids, hist_eids, hist_ts, hist_dirs = self.graph.get_history(
                nids.cpu().numpy(), ts.cpu().numpy(), self.hist_len)
            anonymized_ids = anonymized_reindex(hist_nids)
            hist_nids = torch.from_numpy(hist_nids).to(device).long()  # [bs, len]
            anonymized_ids = torch.from_numpy(anonymized_ids).to(device).long()
            hist_eids = torch.from_numpy(hist_eids).to(device).long()
            hist_ts = torch.from_numpy(hist_ts).to(device).float()
            hist_dirs = torch.from_numpy(hist_dirs).to(device).long()
        else:
            data = computation_graph.restart_data
            hist_nids = data.hist_nids
            anonymized_ids = data.anonymized_ids
            hist_eids = data.hist_eids
            hist_ts = data.hist_ts
            hist_dirs = data.hist_dirs

        bs, hist_len = hist_nids.shape
        mask = (hist_nids == 0)
        mask[:, -1] = False
        invalid_rows = mask.all(1, keepdims=True)

        r_nids = nids.unsqueeze(1).repeat(1, hist_len)
        src_nids = r_nids * hist_dirs + hist_nids * (1-hist_dirs)
        dst_nids = r_nids * (1-hist_dirs) + hist_nids * hist_dirs

        src_vals = self.raw_feat_getter.get_node_embeddings(src_nids)
        dst_vals = self.raw_feat_getter.get_node_embeddings(dst_nids)
        edge_vals = self.raw_feat_getter.get_edge_embeddings(hist_eids)
        anony_vals = self.anony_emb(anonymized_ids)
        ts_vals = self.time_encoder(hist_ts[:, -1].unsqueeze(1) - hist_ts)
        full_vals = torch.cat([src_vals, dst_vals, anony_vals, edge_vals, ts_vals], 2)

        last_event_feat = full_vals[:, -1, :self.d_model - self.tfeat_dim]
        full_vals[:, -1, :self.d_model - self.tfeat_dim] = 0.
        qkv = full_vals.transpose(0, 1)
        out, _ = self.mha_fn(qkv, qkv, qkv, key_padding_mask=mask)

        # h_prev_left = self.out_fn(F.relu(out.mean(0)))
        involved_node_reprs = hist_eids
        node_ids = hist_nids
        h_prev_left = self.temporal_embedding_fn.compute_embedding_with_computation_graph(
            involved_node_reprs, node_ids, ts_vals, computation_graph, self.neural_process
        )

        h_prev_right = self.jump_update(last_event_feat, h_prev_left)

        prev_prev_ts = hist_ts[:, -2]
        prev_ts = hist_ts[:, -1]

        h_prev_left = self.ode_transform(h_prev_right, prev_prev_ts.unsqueeze(-1) / resize_ratio_st, prev_ts.unsqueeze(-1) / resize_ratio_st)

        h_prev_left = h_prev_left.masked_fill(invalid_rows, 0.)
        h_prev_right = h_prev_right.masked_fill(invalid_rows, 0.)

        return h_prev_left, h_prev_right, prev_ts


class WalkRestarter(Restarter):
    def __init__(self, raw_feat_getter: FeatureGetter, graph: Graph,
                 *, n_walks: int=20, walk_length: int=5, alpha=1e-5,
                 n_head=2, dropout=0.1):
        super().__init__(raw_feat_getter, graph)
        self.n_walks = n_walks
        self.walk_length = walk_length
        self.alpha = alpha
        self.n_head = n_head
        self.dropout = dropout

        self.anony_emb = nn.Sequential(
            nn.Linear(walk_length, self.nfeat_dim),
            nn.ReLU(),
            nn.Linear(self.nfeat_dim, self.nfeat_dim)
        )

        self.d_model = self.nfeat_dim * 2 + self.efeat_dim + self.tfeat_dim
        self.d_last_edge = self.nfeat_dim * 4 + self.efeat_dim
        self.seq_mha_fn = nn.MultiheadAttention(self.d_model, n_head, dropout)
        self.agg_mha_fn = nn.MultiheadAttention(self.d_model, n_head, dropout)
        self.out_fn = nn.Linear(self.d_model, self.nfeat_dim)
        self.merger = MergeLayer(self.nfeat_dim, self.d_last_edge, 
                                 self.nfeat_dim, self.nfeat_dim, dropout=dropout)

    def forward(self, nids: Tensor, ts: Tensor,
                computation_graph: Optional[ComputationGraph]=None
               ) -> Tuple[Tensor, Tensor, Tensor]:

        if computation_graph is None:
            device = nids.device
            np_nids = nids.cpu().numpy()
            np_ts = ts.cpu().numpy()
            prev_neighbors, prev_eids, prev_ts, prev_dirs = (
                x.squeeze(1) for x in self.graph.get_history(np_nids, np_ts, 1))
            walk_nids, walk_eids, walk_ts = self.graph.sample_walks(
                np_nids, prev_ts, self.n_walks, self.walk_length, self.alpha
            )
            prev_srcs = (1-prev_dirs) * np_nids + prev_dirs * prev_neighbors
            prev_dsts = prev_dirs * np_nids + (1-prev_dirs) * prev_neighbors
            prev_srcs[prev_neighbors == 0] = 0
            prev_dsts[prev_neighbors == 0] = 0

            walk_anonymized_codes, id2code_dicts = set_anonymized_encoding(walk_nids)

            prev_srcs_codes = np.zeros((len(nids), self.walk_length))
            prev_dsts_codes = np.zeros((len(nids), self.walk_length))
            for i in range(len(nids)):
                prev_srcs_codes[i] = id2code_dicts[i].get(prev_srcs[i], 0)
                prev_dsts_codes[i] = id2code_dicts[i].get(prev_dsts[i], 0)

            prev_srcs = torch.from_numpy(prev_srcs).long().to(device)
            prev_dsts = torch.from_numpy(prev_dsts).long().to(device)
            prev_eids = torch.from_numpy(prev_eids).long().to(device)
            walk_nids = torch.from_numpy(walk_nids).long().to(device)
            walk_anonymized_codes = torch.from_numpy(walk_anonymized_codes).float().to(device)
            walk_eids = torch.from_numpy(walk_eids).long().to(device)
            walk_ts = torch.from_numpy(walk_ts).float().to(device)
            prev_srcs_codes = torch.from_numpy(prev_srcs_codes).float().to(device)
            prev_dsts_codes = torch.from_numpy(prev_dsts_codes).float().to(device)
        else:
            data = computation_graph.restart_data
            prev_srcs = data.prev_srcs
            prev_dsts = data.prev_dsts
            prev_srcs_codes = data.prev_srcs_codes
            prev_dsts_codes = data.prev_dsts_codes
            prev_eids = data.prev_eids
            walk_nids = data.walk_nids
            walk_anonymized_codes = data.walk_anonymized_codes
            walk_eids = data.walk_eids
            walk_ts = data.walk_ts

        bs, n_walks, walk_length = walk_nids.shape

        prev_ts = walk_ts[:, 0, -1]

        walk_nids = walk_nids.reshape(bs * n_walks, walk_length)
        walk_eids = walk_eids.reshape(bs * n_walks, walk_length)
        walk_ts = walk_ts.reshape(bs * n_walks, walk_length)
        walk_anonymized_codes = walk_anonymized_codes.reshape(bs * n_walks, walk_length, walk_length)

        node_vals = self.raw_feat_getter.get_node_embeddings(walk_nids)
        edge_vals = self.raw_feat_getter.get_edge_embeddings(walk_eids)
        anony_vals = self.anony_emb(walk_anonymized_codes)
        ts_vals = self.time_encoder(walk_ts[:, -1:] - walk_ts)

        full_vals = torch.cat([node_vals, edge_vals, anony_vals, ts_vals], 2)
        mask = (walk_nids == 0)
        mask[:, -1] = False

        qkv = full_vals.transpose(0, 1)
        walk_reprs, _ = self.seq_mha_fn(qkv, qkv, qkv, key_padding_mask=mask)
        walk_reprs = walk_reprs.mean(0).reshape(bs, n_walks, self.d_model).transpose(0, 1)
        agg_reprs, _ = self.agg_mha_fn(walk_reprs, walk_reprs, walk_reprs)
        agg_reprs = agg_reprs.mean(0)

        h_prev_left = self.out_fn(F.relu(agg_reprs))

        last_event_feat = self.get_edge_reprs(prev_srcs, prev_dsts,
                                              prev_srcs_codes, prev_dsts_codes, 
                                              prev_eids)

        h_prev_right = self.merger(h_prev_left, last_event_feat)
        invalid_rows = (prev_srcs == 0).unsqueeze(1)
        h_prev_left = h_prev_left.masked_fill(invalid_rows, 0.)
        h_prev_right = h_prev_right.masked_fill(invalid_rows, 0.)

        return h_prev_left, h_prev_right, prev_ts

    def get_edge_reprs(self, srcs, dsts, srcs_codes, dsts_codes, eids):
        bs = len(srcs)
        nfeats = self.raw_feat_getter.get_node_embeddings(
            torch.stack([srcs, dsts], dim=1)
        ).reshape(bs, 2 * self.nfeat_dim)
        efeats = self.raw_feat_getter.get_edge_embeddings(eids)
        anony_codes = self.anony_emb(
            torch.stack([srcs_codes, dsts_codes], dim=1)
        ).reshape(bs, 2 * self.nfeat_dim)
        full_reprs = torch.cat([nfeats, efeats, anony_codes], dim=1)
        return full_reprs


class StaticRestarter(Restarter):
    def __init__(self, raw_feat_getter: FeatureGetter, graph: Graph):
        super().__init__(raw_feat_getter, graph)
        self.left_emb = nn.Embedding(self.n_nodes, self.nfeat_dim)
        self.right_emb = nn.Embedding(self.n_nodes, self.nfeat_dim)
        nn.init.zeros_(self.left_emb.weight)
        nn.init.zeros_(self.right_emb.weight)
    
    def forward(self, nids: Tensor, ts: Tensor, 
                computation_graph: Optional[ComputationGraph]=None
               ) -> Tuple[Tensor, Tensor, Tensor]:
        if computation_graph is None:
            device = nids.device
            _, _, prev_ts, _ = self.graph.get_history(
                nids.cpu().numpy(), ts.cpu().numpy(), 1)
            prev_ts = prev_ts[:, 0]
            prev_ts = torch.from_numpy(prev_ts).to(device).float()
        else:
            data = computation_graph.restart_data
            prev_ts = data.prev_ts
        h_left = self.left_emb(nids)
        h_right = self.right_emb(nids)

        return h_left, h_right, prev_ts
