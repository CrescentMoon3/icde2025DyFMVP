import math
from typing import Optional, Tuple, Union
import warnings

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.data_loader import ComputationGraph
from ..data.graph import Graph
from .basic_modules import MergeLayer
from .feature_getter import FeatureGetter, NumericalFeature
from .memory import (Memory, MessageStore, MessageStoreNoGrad,
                     MessageStoreNoGradLastOnly)
from .message_modules import (IdentityMessageFunction, LastMessageAggregator,
                              LastMessageAggregatorNoGrad,
                              LastMessageAggregatorNoGradLastOnly,
                              LinearMessageFunction, MLPMessageFunction)
from .temporal_agg_modules import GraphAttnEmbedding

from .time_encoding import TimeEncode as TimeEncode_ori

from .update_modules import GRUUpdater, MergeUpdater
from .model_utils import select_latest_nids
from .restarters import Restarter, SeqRestarter, TaroRestarter

# add dependency of CNNDP
from .modules import GeneralModel
from .model_utils import ContextTargetSpliter, parse_config
from torch.distributions import Normal
from .layers import EdgePredictor, TimeEncode, SineActivation
from torchdiffeq import odeint_adjoint

import pandas as pd
import pathlib


class FMVP(nn.Module):
    def __init__(self, *, raw_feat_getter: FeatureGetter, graph: Graph,
                 n_neighbors: int=20, n_layers: int=2, n_head: int=2, dropout: float=0.1,
                 msg_src: str, upd_src: str,
                 msg_tsfm_type: str='id', mem_update_type: str='gru',
                 tgn_mode: bool=True, msg_last_only: bool=True,
                 hit_type: str='none',
                 data_name: str):

        super().__init__()
        self.raw_feat_getter = raw_feat_getter
        self.n_nodes = self.raw_feat_getter.n_nodes
        self.nfeat_dim = self.raw_feat_getter.nfeat_dim
        self.efeat_dim = self.raw_feat_getter.efeat_dim

        config_path = "config/CMNDP_{}.yml".format(data_name)
        print(config_path)
        
        sample_param, memory_param, gnn_param, np_param, train_param = parse_config(config_path)

        # time_enc_dim: origin (172/54)
        self.time_encoder = TimeEncode_ori(dim=self.nfeat_dim)

        self.tfeat_dim = self.time_encoder.dim
        self.memory_dim = self.nfeat_dim
        self.raw_msg_dim = self.memory_dim * 2 + self.efeat_dim + self.tfeat_dim
        self.msg_dim = None

        self.n_neighbors = n_neighbors
        self.n_layers = n_layers

        self.msg_src = msg_src
        self.upd_src = upd_src

        self.tgn_mode = True if msg_last_only else tgn_mode
        self.msg_last_only = msg_last_only

        self.left_memory = Memory(self.n_nodes, self.memory_dim)
        self.right_memory = Memory(self.n_nodes, self.memory_dim)
        if self.msg_last_only:
            self.msg_store = MessageStoreNoGradLastOnly(self.n_nodes, dim=self.raw_msg_dim)
        elif self.tgn_mode:
            self.msg_store = MessageStoreNoGrad(self.n_nodes)
        else:
            self.msg_store = MessageStore(self.n_nodes)
        self.msg_memory = self.left_memory if self.msg_src == 'left' else self.right_memory
        self.upd_memory = self.left_memory if self.upd_src == 'left' else self.right_memory

        if self.msg_last_only:
            self.msg_aggregate_fn = LastMessageAggregatorNoGradLastOnly(
                raw_feat_getter=self.raw_feat_getter,
                time_encoder=self.time_encoder,
            )
        elif self.tgn_mode:
            self.msg_aggregate_fn = LastMessageAggregatorNoGrad(
                raw_feat_getter=self.raw_feat_getter,
                time_encoder=self.time_encoder,
            )
        else:
            self.msg_aggregate_fn = LastMessageAggregator(
                raw_feat_getter=self.raw_feat_getter,
                time_encoder=self.time_encoder,
            )

        if msg_tsfm_type == 'id':
            self.msg_transform_fn = IdentityMessageFunction(raw_msg_dim=self.raw_msg_dim)
        elif msg_tsfm_type == 'linear':
            self.msg_transform_fn = LinearMessageFunction(raw_msg_dim=self.raw_msg_dim)
        elif msg_tsfm_type == 'mlp':
            self.msg_transform_fn = MLPMessageFunction(raw_msg_dim=self.raw_msg_dim)
        else:
            raise NotImplementedError
        self.msg_dim = self.msg_transform_fn.output_size

        if mem_update_type == 'gru':
            self.right_mem_updater = GRUUpdater(self.msg_dim, self.memory_dim)
        elif mem_update_type == 'merge':
            self.right_mem_updater = MergeUpdater(self.msg_dim, self.memory_dim)
        elif mem_update_type == 'msa':
            self.right_mem_updater = jump_attn_agger(self.memory_dim,
                                                    self.msg_dim,
                                                    n_head,
                                                    dropout)
        else:
            raise NotImplementedError

        self.temporal_embedding_fn = GraphAttnEmbedding(
            raw_feat_getter=self.raw_feat_getter, 
            time_encoder=self.time_encoder, graph=graph, 
            n_neighbors=n_neighbors, n_layers=n_layers, n_head=n_head, 
            dropout=dropout
        )

        config_path = "config/CMNDP_{}.yml".format(data_name)
        root = pathlib.Path('.')
        name = data_name
        print('data_name: '+str(name))
        graph_df = pd.read_csv(root / 'data/ml_{}.csv'.format(name))
        ts = graph_df.ts.values
        print(ts.max())
        resize_ratio = 10 ** (len(str(int(ts.max()))) - 1)
        print(f"Resize ratio is automatically set to: {resize_ratio}")

        combine_first = False
        if 'combine_neighs' in train_param and train_param['combine_neighs']:
            combine_first = True
        
        self.train_neg_samples = train_param.get('train_neg_size', 1)

        self.neural_process = NeuralProcess(self.raw_feat_getter.nfeat_dim, self.raw_feat_getter.efeat_dim, sample_param, memory_param, gnn_param, train_param, np_param,
                      'snp', True, False, resize_ratio,
                      self.device, combined=combine_first).to(self.device)

        self.hit_type = hit_type
        if self.hit_type == 'vec':
            merge_dim = self.nfeat_dim + self.n_neighbors
        elif self.hit_type == 'bin':
            self.hit_embedding = nn.Embedding(2, self.nfeat_dim)
            merge_dim = self.nfeat_dim
        elif self.hit_type == 'count':
            self.hit_embedding = nn.Embedding(self.n_neighbors + 1, self.nfeat_dim)
            merge_dim = self.nfeat_dim
        else:
            merge_dim = self.nfeat_dim

        self.score_fn = MergeLayer(merge_dim, merge_dim,
                                   self.nfeat_dim, 1, dropout=dropout)

        self.contrast_loss_fn = nn.BCEWithLogitsLoss()

        self._sanity_check()
    
    def _sanity_check(self):
        if self.msg_src not in {'left', 'right'}:
            raise ValueError(f'Invalid msg_src={self.msg_src}')
        if self.upd_src not in {'left', 'right'}:
            raise ValueError(f'Invalid upd_src={self.msg_src}')
    
    @property
    def graph(self):
        return self.temporal_embedding_fn.graph
    
    @graph.setter
    def graph(self, new_obj: Graph):
        self.temporal_embedding_fn.graph = new_obj
    
    @property
    def device(self):
        return self.msg_memory.device

    def contrast_learning(self, src_ids: Tensor, dst_ids: Tensor, neg_dst_ids: Tensor,
                         ts: Tensor, eids: Tensor,
                         computation_graph: ComputationGraph
                         ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        bs = len(src_ids)
        pos_node_ids = torch.cat([src_ids, dst_ids])
        batch_node_ids = torch.cat([src_ids, dst_ids, neg_dst_ids])

        outdated_nids, msgs, prev_ts = self.compute_messages(
            computation_graph.np_computation_graph_nodes)

        involved_node_ids = computation_graph.computation_graph_nodes
        involved_node_reprs = self.right_memory.vals[involved_node_ids].clone()
        if len(outdated_nids):
            h_prev_right_all = self.apply_messages(outdated_nids, msgs, prev_ts)  # h(t'+)
            involved_local_index, outdated_local_index = torch.where(
                involved_node_ids[:, None] == outdated_nids)
            involved_node_reprs[involved_local_index] = h_prev_right_all[outdated_local_index]

        local_center_nids = computation_graph.local_index[batch_node_ids]
        center_node_reprs = involved_node_reprs[local_center_nids]

        # TODO: verify NP+NODE in stage 2

        batch_node_ids_agg = torch.tensor(list(batch_node_ids)).to(self.device)
        prev_t_batch_node = self.msg_memory.update_ts[batch_node_ids_agg]

        if self.training:
            h_left_with_negs_np, q_target, q_context = self.neural_process(center_node_reprs, prev_t_batch_node, ts.repeat(3), self.train_neg_samples)
        else:
            h_left_with_negs_np = self.neural_process(center_node_reprs, prev_t_batch_node, ts.repeat(3), self.train_neg_samples)

        involved_node_reprs[local_center_nids] = h_left_with_negs_np
        h_left_with_negs = self.compute_temporal_embedding_with_involved_nodes_only(
            involved_node_reprs, batch_node_ids, ts.repeat(3), computation_graph
        )

        if len(outdated_nids):
            unique_pos_ids, _ = select_latest_nids(pos_node_ids, ts.repeat(2))
            pos_index, mem_index = torch.where(unique_pos_ids[:, None] == outdated_nids)
            if len(pos_index):
                outdated_pos_ids = unique_pos_ids[pos_index]
                update_vals = h_prev_right_all.detach()[mem_index]
                update_prev_ts = prev_ts[mem_index]
                self.msg_store.clear(outdated_pos_ids)
                self.update_right_memory(outdated_pos_ids, update_vals, update_prev_ts)

        self.store_events(src_ids, dst_ids, ts, eids)

        h_prev_left, _ = self.left_memory.get(pos_node_ids)
        h_prev_right, _ = self.right_memory.get(pos_node_ids)
        h_prev_left = h_prev_left.clone()
        h_prev_right = h_prev_right.clone()

        h_left = h_left_with_negs[:2*bs]
        self.update_left_memory(pos_node_ids, h_left, ts.repeat(2))

        x, y, neg_y = h_left_with_negs.reshape(3, bs, self.nfeat_dim)
        src_hit, dst_hit, neg_src_hit, neg_dst_hit = computation_graph.hit_data
        if self.hit_type == 'vec':
            x_pos_pair = torch.cat([x, src_hit], 1)
            y_pos_pair = torch.cat([y, dst_hit], 1)
            x_neg_pair = torch.cat([x, neg_src_hit], 1)
            y_neg_pair = torch.cat([neg_y, neg_dst_hit], 1)
        elif self.hit_type == 'bin':
            x_pos_pair = x + self.hit_embedding(src_hit.max(1).values.long())
            y_pos_pair = y + self.hit_embedding(dst_hit.max(1).values.long())
            x_neg_pair = x + self.hit_embedding(neg_src_hit.max(1).values.long())
            y_neg_pair = neg_y + self.hit_embedding(neg_dst_hit.max(1).values.long())
        elif self.hit_type == 'count':
            x_pos_pair = x + self.hit_embedding(src_hit.sum(1).long())
            y_pos_pair = y + self.hit_embedding(dst_hit.sum(1).long())
            x_neg_pair = x + self.hit_embedding(neg_src_hit.sum(1).long())
            y_neg_pair = neg_y + self.hit_embedding(neg_dst_hit.sum(1).long())
        else:
            x_pos_pair = x_neg_pair = x
            y_pos_pair = y
            y_neg_pair = neg_y

        pos_scores = self.score_fn(x_pos_pair, y_pos_pair).squeeze(1)
        neg_scores = self.score_fn(x_neg_pair, y_neg_pair).squeeze(1)

        label_ones = torch.ones_like(pos_scores)
        label_zeros = torch.zeros_like(neg_scores)
        labels = torch.cat([label_ones, label_zeros], 0)
        contrast_loss = self.contrast_loss_fn(
            torch.cat([pos_scores, neg_scores], 0), labels)

        if self.training:
            return contrast_loss, h_left, pos_scores, neg_scores, h_prev_left, h_prev_right, q_target, q_context
        else:
            return contrast_loss, h_left, pos_scores, neg_scores, h_prev_left, h_prev_right

    
    def compute_messages(self, node_ids: Union[Tensor, np.ndarray, None]=None
                        ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        outdated_nodes = self.msg_store.get_outdated_node_ids(node_ids)
        outdated_nodes = outdated_nodes.to(self.device)

        if len(outdated_nodes) == 0:
            return (outdated_nodes, None, None)

        last_update_ts = self.msg_memory.update_ts[outdated_nodes]
        raw_msgs, ts = self.msg_aggregate_fn(outdated_nodes, last_update_ts, self.msg_store.node_messages)

        if self.msg_src == 'left' and not (ts == last_update_ts).all().item():
            raise ValueError("Messages' ts should be equal to last update ts "
                             "when using left memory as msg source.")

        if self.tgn_mode:
            raw_msgs = raw_msgs.detach()
        msgs = self.msg_transform_fn(raw_msgs)
        return outdated_nodes, msgs, ts

    def apply_messages(self, node_ids: Tensor, msgs: Tensor, ts: Tensor) -> Tensor:
        old_vals, last_update_ts = self.upd_memory.get(node_ids)
        delta_ts = ts - last_update_ts
        new_vals = self.right_mem_updater(old_vals, msgs, delta_ts)

        return new_vals
    
    def compute_temporal_embedding_with_involved_nodes_only(
            self, involved_node_reprs: Tensor, node_ids: Tensor, ts: Tensor,
            computation_graph: ComputationGraph
        ) -> Tensor:

        h = self.temporal_embedding_fn.compute_embedding_with_computation_graph(
            involved_node_reprs, node_ids, ts, computation_graph, self.neural_process
        )
        return h

    def temporal_embedding(self, memory: Memory, node_ids: Tensor, ts: Tensor) -> Tensor:

        warnings.warn('Please use "compute_embedding_with_computation_graph" instead!',
                      DeprecationWarning)
        h = self.temporal_embedding_fn.compute_embedding(all_node_reprs=memory.vals,
                                                         np_center_nids=node_ids.cpu().numpy(),
                                                         np_ts=ts.cpu().numpy())
        return h

    @torch.no_grad()
    def update_right_memory(self, node_ids: Tensor, new_vals: Tensor, ts: Tensor):

        self.right_memory.set(node_ids, new_vals, ts)
    
    @torch.no_grad()
    def update_left_memory(self, node_ids: Tensor, new_vals: Tensor, ts: Tensor):

        node_ids, index = select_latest_nids(node_ids, ts)
        self.left_memory.set(node_ids, new_vals[index], ts[index])

    @torch.no_grad()
    def store_events(self, src_ids: Tensor, dst_ids: Tensor, ts: Tensor, eids: Tensor):

        src_vals, src_prev_ts = self.msg_memory.get(src_ids)
        dst_vals, dst_prev_ts = self.msg_memory.get(dst_ids)

        if (src_prev_ts > ts).any().item() or (dst_prev_ts > ts).any().item():
            raise ValueError('Events occur before the udpated memory.')

        self.msg_store.store_events(src_ids, dst_ids, src_prev_ts, dst_prev_ts, 
                                    src_vals, dst_vals, eids, ts,
                                    self.raw_feat_getter, self.time_encoder)
    
    @torch.no_grad()
    def flush_msg(self):

        outdated_nids, msgs, prev_ts = self.compute_messages()
        if len(outdated_nids):
            h_prev_right = self.apply_messages(outdated_nids, msgs, prev_ts)
            _ = self.update_right_memory(outdated_nids, h_prev_right, prev_ts)
            self.msg_store.clear(outdated_nids)

    def reset(self):

        self.left_memory.clear()
        self.right_memory.clear()
        self.msg_store.clear()
    
    def save_memory_state(self) -> Tuple[Memory, Memory, MessageStore]:

        left_memory = self.left_memory.clone()
        right_memory = self.right_memory.clone()
        msg_store = self.msg_store.clone()
        data = (left_memory, right_memory, msg_store)
        return data

    def load_memory_state(self, data: Tuple[Memory, Memory, MessageStore]):

        (left_memory, right_memory, msg_store) = data
        self.left_memory = left_memory
        self.right_memory = right_memory
        self.msg_memory = self.left_memory if self.msg_src == 'left' else self.right_memory
        self.upd_memory = self.left_memory if self.upd_src == 'left' else self.right_memory
        self.msg_store = msg_store

    def _get_computation_graph_nodes(
            self, nids: np.ndarray, ts: np.ndarray, depth: Optional[int]=None
        ) -> np.ndarray:

        warnings.warn('This method is no longer useful!',
                      DeprecationWarning)
        depth = self.n_layers if depth is None else depth
        if depth == 0:
            return np.unique(nids)
        ngh_nids, _, neigh_ts, *_ = self.graph.sample_temporal_neighbor(nids, ts, self.n_neighbors)
        r_nids = self._get_computation_graph_nodes(ngh_nids.flatten(), neigh_ts.flatten(), depth-1)
        return np.unique(np.concatenate([nids, r_nids]))


class DYFMVP(FMVP):

    def __init__(self, *, raw_feat_getter: FeatureGetter, graph: Graph, restarter: Restarter,
                 n_neighbors: int=10, n_layers: int=1,
                 n_head: int=2, dropout: float=0.1,
                 msg_src: str, upd_src: str,
                 msg_tsfm_type: str='id', mem_update_type: str='msa',
                 tgn_mode: bool=True, msg_last_only: bool=True,
                 hit_type: str='vec',
                 data_name: str):

        super().__init__(
            raw_feat_getter=raw_feat_getter, graph=graph,
            n_neighbors=n_neighbors, n_layers=n_layers, n_head=n_head, dropout=dropout,
            msg_src=msg_src, upd_src=upd_src,
            msg_tsfm_type=msg_tsfm_type, mem_update_type=mem_update_type,
            tgn_mode=tgn_mode, msg_last_only=msg_last_only,
            hit_type=hit_type,
            data_name=data_name
        )
        self.restarter_fn = restarter
        self.mutual_loss_fn = nn.MSELoss()

        name = data_name
        root = pathlib.Path('.')
        graph_df = pd.read_csv(root / 'data/ml_{}.csv'.format(name))
        ts = graph_df.ts.values
        self.resize_ratio_st = 10 ** (len(str(int(ts.max()))) - 1)
                                
    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:

        return self.contrast_and_mutual_learning(*args, **kwargs)
    
    def contrast_and_mutual_learning(
            self, src_ids: Tensor, dst_ids: Tensor, neg_dst_ids: Tensor,
            ts: Tensor, eids: Tensor, computation_graph: ComputationGraph,
            contrast_only: bool=False
        ) -> Tuple[Tensor, Tensor]:

        if self.training:
            contrast_loss, *_, h_prev_left, h_prev_right, q_target, q_context = self.contrast_learning(
            src_ids, dst_ids, neg_dst_ids, ts, eids, computation_graph)
        else:
            contrast_loss, *_, h_prev_left, h_prev_right = self.contrast_learning(
            src_ids, dst_ids, neg_dst_ids, ts, eids, computation_graph)

        if contrast_only:
            mutual_loss = torch.tensor(0, device=contrast_loss.device)
            if self.training:
                return contrast_loss, mutual_loss, q_target, q_context
            else:
                return contrast_loss, mutual_loss

        index = computation_graph.restart_data.index
        unique_nids = torch.cat([src_ids, dst_ids])[index]
        unique_ts = ts.repeat(2)[index]

        if type(self.restarter_fn) == TaroRestarter:
            prev_ts_restart = self.restarter_fn(
            unique_nids, unique_ts, computation_graph, 0, self.resize_ratio_st)

            unique_prev_ts = prev_ts_restart
            surrogate_h_prev_left, surrogate_h_prev_right, _ = self.restarter_fn(
            unique_nids, unique_prev_ts, computation_graph, 1, self.resize_ratio_st)

        else: 
            surrogate_h_prev_left, surrogate_h_prev_right, _ = self.restarter_fn(
            unique_nids, unique_ts, computation_graph)

        targets = torch.cat([h_prev_left[index], h_prev_right[index]], 0)
        preds = torch.cat([surrogate_h_prev_left, surrogate_h_prev_right], 0)
        valid_rows = torch.where(~(targets == 0).all(1))[0]

        if len(valid_rows):
            mutual_loss = self.mutual_loss_fn(preds[valid_rows], targets[valid_rows].detach())
        else:
            mutual_loss = torch.tensor(0, device=contrast_loss.device)

        if self.training:
            return contrast_loss, mutual_loss, q_target, q_context
        else:
            return contrast_loss, mutual_loss

    @torch.no_grad()
    def restart(self, nids: Tensor, ts: Tensor, mix: float=0.):

        if len(nids):
            self.msg_store.clear(nids)
            if type(self.restarter_fn) == TaroRestarter:
                prev_ts_restart = self.restarter_fn(nids, ts, None, 0, self.resize_ratio_st)

                unique_prev_ts = prev_ts_restart
                h_prev_left, h_prev_right, prev_ts = self.restarter_fn(nids, unique_prev_ts, None, 1, self.resize_ratio_st)
            
            else:
                h_prev_left, h_prev_right, prev_ts = self.restarter_fn(nids, ts)
            if mix > 0:
                h_prev_left = mix * h_prev_left + (1-mix) * self.left_memory.vals[nids]
                h_prev_right = mix * h_prev_right + (1-mix) * self.right_memory.vals[nids]
            self.left_memory.set(nids, h_prev_left, prev_ts, skip_check=True)
            self.right_memory.set(nids, h_prev_right, prev_ts, skip_check=True)

    @property
    def graph(self):
        return self.temporal_embedding_fn.graph

    @graph.setter
    def graph(self, new_obj: Graph):
        self.temporal_embedding_fn.graph = new_obj
        self.restarter_fn.graph = new_obj

class NeuralProcess(GeneralModel):
    def __init__(self, gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, np_parm,
                 base_model, enable_ode, enable_determinstic, resize_ratio, device,
                 combined=False):
        super(NeuralProcess, self).__init__(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param,
                                            train_param,
                                            combined=combined)
        self.base_model = base_model
        self.enable_ode = enable_ode
        self.enabe_determinstic = enable_determinstic
        self.r_dim = np_parm['r_dim']
        self.z_dim = np_parm['z_dim']
        self.h_dim = np_parm['h_dim']
        # self.t_dim = gnn_dim_node
        self.t_dim = np_parm['t_dim']
        self.np_out_dim = np_parm['out_dim']
        self.l = np_parm['l']
        self.old_as_context = np_parm['old_as_context']
        self.r_tol = float(np_parm['r_tol'])
        self.a_tol = float(np_parm['a_tol'])
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.method = np_parm['method']
        self.resize_ratio = resize_ratio

        self.encoder = Encoder(
            # gnn_param['dim_out'] * 2, np_parm['y_dim'], self.h_dim, self.r_dim
            # (gnn_param['dim_out'] + np_parm['t_dim'] * 3) * 2, np_parm['y_dim'], self.h_dim, self.r_dim
            # (gnn_param['dim_out'] + np_parm['t_dim']) * 2, np_parm['y_dim'], self.h_dim, self.r_dim
            (gnn_param['dim_out']) * 2, np_parm['y_dim'], self.h_dim, self.r_dim
            )
        self.r_to_mu_sigma = MuSigmaEncoder(self.r_dim, self.z_dim)
        self.decoder = Decoder(
            # gnn_param['dim_out'], 
            # gnn_param['dim_out'] + np_parm['t_dim'] * 3,
            # gnn_param['dim_out'] + np_parm['t_dim'],
            gnn_param['dim_out'],
            self.z_dim,
            self.h_dim,
            self.np_out_dim)
        self.time_encoder = TimeEncode(np_parm['t_dim'])
        if self.enable_ode:
            self.ode_solver = ODESolver(self.r_dim, self.h_dim, self.t_dim, self.time_encoder, r_tol=self.r_tol,
                                        a_tol=self.a_tol,
                                        method=self.method)
        if not self.old_as_context:
            self.context_spliter = ContextTargetSpliter(
                np_parm['context_split'])
        if self.base_model == 'snp':
            self.update_cell = nn.GRUCell(self.r_dim, self.r_dim)
        elif self.base_model == "anp":
            self.update_cell = nn.GRUCell(self.r_dim, self.r_dim)
            self.multi_atten = nn.MultiheadAttention(self.r_dim, num_heads=4, kdim=self.r_dim,
                                                     vdim=self.r_dim)
            self.history_memory = torch.zeros(
                1, self.r_dim, device=self.device)
        elif self.base_model == "mnp":
            self.memory_net = MemoryNet(np_parm['mem_size'], self.h_dim, self.t_dim, self.r_dim, self.time_encoder,
                                        device=self.device)
        if self.enabe_determinstic:
            self.deterministic_decoder = DeterminsticDecoder(gnn_param['dim_out'], self.r_dim, self.h_dim,
                                                             self.np_out_dim)
        self.register_buffer('running_r', torch.zeros(self.r_dim, device=self.device))
        self.register_buffer('num_batches_tracked', torch.tensor(
            0, dtype=torch.long, device=self.device))
        self.register_buffer('last_ts', torch.tensor(
            0, dtype=torch.float, device=self.device))
        self.test = False

        self.merge_aug = MergeAUG(
            (gnn_param['dim_out']) * 2, gnn_param['dim_out'], dropout=train_param['dropout']
        )

    def detach(self):
        self.running_r = self.running_r.detach()
        if self.base_model == "anp":
            self.history_memory.detach_()
        elif self.base_model == 'mnp':
            self.memory_net.memory.detach_()

    def reset(self):

        self.running_r = torch.zeros(self.r_dim, device=self.device)
        self.num_batches_tracked = torch.tensor(
            0, dtype=torch.long, device=self.device)
        self.last_ts = torch.tensor(0, dtype=torch.float, device=self.device)
        if self.base_model == 'mnp':
            self.memory_net.reset()
        elif self.base_model == 'anp':
            self.history_memory = torch.zeros(
                1, self.r_dim, device=self.device)

    def nodes_np_drift_transform(self, h, prev_ts, ts, neg_samples=1):

        ts = ts / self.resize_ratio
        prev_ts = prev_ts / self.resize_ratio

        num_edge = h.shape[0] // (neg_samples + 2)

        # split h, prev_ts, ts
        src = h[:num_edge]
        ts_src = ts[:num_edge]
        pos_dst = h[num_edge:2 * num_edge]
        ts_pos_dst = ts[num_edge:2 * num_edge]
        neg_dst = h[2 * num_edge:]
        ts_neg_dst = ts[2 * num_edge:]

        prev_ts_src = prev_ts[:num_edge]
        prev_ts_pos_dst = prev_ts[num_edge:2 * num_edge]
        prev_ts_neg_dst = prev_ts[2 * num_edge:]
        src_hid = src
        pos_dst_hid = pos_dst
        neg_dst_hid = neg_dst

        delta_ts_src_emb = self.time_encoder((ts_src[:, None]-prev_ts_src).view(-1))
        delta_ts_pos_dst_emb = self.time_encoder((ts_pos_dst[:, None]-prev_ts_pos_dst).view(-1))
        delta_ts_neg_dst_emb = self.time_encoder((ts_neg_dst[:, None]-prev_ts_neg_dst).view(-1))

        prev_ts_src_emb = self.time_encoder(prev_ts_src.view(-1))
        prev_ts_pos_dst_emb = self.time_encoder(prev_ts_pos_dst.view(-1))
        prev_ts_neg_dst_emb = self.time_encoder(prev_ts_neg_dst.view(-1))

        ts_src_emb = self.time_encoder(ts_src.unsqueeze(1).expand(-1,prev_ts_src.shape[1]).contiguous().view(-1))
        ts_pos_dst_emb = self.time_encoder(ts_pos_dst.unsqueeze(1).expand(-1,prev_ts_src.shape[1]).contiguous().view(-1))
        ts_neg_dst_emb = self.time_encoder(ts_neg_dst.unsqueeze(1).expand(-1,prev_ts_src.shape[1]).contiguous().view(-1))

        ts_src_fin = delta_ts_src_emb + prev_ts_src_emb + ts_src_emb
        ts_pos_dst_fin = delta_ts_pos_dst_emb + prev_ts_pos_dst_emb + ts_pos_dst_emb
        ts_neg_dst_fin = delta_ts_neg_dst_emb + prev_ts_neg_dst_emb + ts_neg_dst_emb

        # src_w_prior = torch.cat([src.view(-1,src.shape[-1]), delta_ts_src_emb, prev_ts_src_emb, ts_src_emb], dim=-1)
        # pos_dst_w_prior = torch.cat([pos_dst.view(-1,pos_dst.shape[-1]), delta_ts_pos_dst_emb, prev_ts_pos_dst_emb, ts_pos_dst_emb], dim=-1)
        # neg_dst_w_prior = torch.cat([neg_dst.view(-1,neg_dst.shape[-1]), delta_ts_neg_dst_emb, prev_ts_neg_dst_emb, ts_neg_dst_emb], dim=-1)
        # src_w_prior = torch.cat([src_hid.view(-1,src_hid.shape[-1]), ts_src_fin], dim=-1)
        # pos_dst_w_prior = torch.cat([pos_dst_hid.view(-1,pos_dst_hid.shape[-1]), ts_pos_dst_fin], dim=-1)
        # neg_dst_w_prior = torch.cat([neg_dst_hid.view(-1,neg_dst_hid.shape[-1]), ts_neg_dst_fin], dim=-1)
        src_w_prior = src_hid.view(-1,src_hid.shape[-1])
        pos_dst_w_prior = pos_dst_hid.view(-1,pos_dst_hid.shape[-1])
        neg_dst_w_prior = neg_dst_hid.view(-1,neg_dst_hid.shape[-1])

        h_w_prior = torch.cat([src_w_prior, pos_dst_w_prior, neg_dst_w_prior], dim=0)

        mu_running, sigma_running = self.r_to_mu_sigma(torch.unsqueeze(self.running_r, 0))

        q_running = Normal(mu_running, sigma_running)
        z_sample = q_running.rsample(torch.Size([self.l]))  # (l, z_dim)

        h__predict = self.decoder(
            h_w_prior, z_sample, ts, self.last_ts, neg_samples=neg_samples
        )

        h__predict = self.merge_aug(h.view(-1,h.shape[-1]), h__predict)

        return h__predict


    def xy_to_mu_sigma(self, data, prev_ts, ts, mode, negative_sample=1):

        src, pos_dst, neg_dst = data
        prev_ts_src, prev_ts_pos_dst, prev_ts_neg_dst = prev_ts
        ts_src, ts_pos_dst, ts_neg_dst = ts
        src_hid = src
        pos_dst_hid = pos_dst
        neg_dst_hid = neg_dst

        delta_ts_src_emb = self.time_encoder(ts_src-prev_ts_src)
        delta_ts_pos_dst_emb = self.time_encoder(ts_pos_dst-prev_ts_pos_dst)
        delta_ts_neg_dst_emb = self.time_encoder(ts_neg_dst-prev_ts_neg_dst)

        prev_ts_src_emb = self.time_encoder(prev_ts_src)
        prev_ts_pos_dst_emb = self.time_encoder(prev_ts_pos_dst)
        prev_ts_neg_dst_emb = self.time_encoder(prev_ts_neg_dst)

        ts_src_emb = self.time_encoder(ts_src)
        ts_pos_dst_emb = self.time_encoder(ts_pos_dst)
        ts_neg_dst_emb = self.time_encoder(ts_neg_dst)

        ts_src_fin = delta_ts_src_emb + prev_ts_src_emb + ts_src_emb
        ts_pos_dst_fin = delta_ts_pos_dst_emb + prev_ts_pos_dst_emb + ts_pos_dst_emb
        ts_neg_dst_fin = delta_ts_neg_dst_emb + prev_ts_neg_dst_emb + ts_neg_dst_emb

        t = torch.cat([ts_src, ts_neg_dst], dim=-1)

        # src_w_prior = torch.cat([src, delta_ts_src_emb, prev_ts_src_emb, ts_src_emb], dim=-1)
        # pos_dst_w_prior = torch.cat([pos_dst, delta_ts_pos_dst_emb, prev_ts_pos_dst_emb, ts_pos_dst_emb], dim=-1)
        # neg_dst_w_prior = torch.cat([neg_dst, delta_ts_neg_dst_emb, prev_ts_neg_dst_emb, ts_neg_dst_emb], dim=-1)
        # src_w_prior = torch.cat([src_hid, ts_src_fin], dim=-1)
        # pos_dst_w_prior = torch.cat([pos_dst_hid, ts_pos_dst_fin], dim=-1)
        # neg_dst_w_prior = torch.cat([neg_dst_hid, ts_neg_dst_fin], dim=-1)
        src_w_prior = src_hid.view(-1,src_hid.shape[-1])
        pos_dst_w_prior = pos_dst_hid.view(-1,pos_dst_hid.shape[-1])
        neg_dst_w_prior = neg_dst_hid.view(-1,neg_dst_hid.shape[-1])

        h_w_prior = torch.cat([src_w_prior, pos_dst_w_prior, neg_dst_w_prior], dim=0)

        pos_pair = torch.cat([src_w_prior, pos_dst_w_prior], dim=-1)
        # neg_pair = torch.cat([src.tile(negative_sample, 1), delta_ts_src_emb.tile(negative_sample, 1), prev_ts_src_emb.tile(negative_sample, 1), ts_src_emb.tile(negative_sample, 1), neg_dst_w_prior], dim=-1)
        neg_pair = torch.cat([src_w_prior, neg_dst_w_prior], dim=-1)
        
        x = torch.cat([pos_pair, neg_pair], dim=0)
        y = torch.cat(
            [torch.ones(pos_pair.shape[0], device=self.device, dtype=torch.long),
             torch.zeros(neg_pair.shape[0], device=self.device, dtype=torch.long)],
            dim=0)

        r_i = self.encoder(x, y)
        r = self.aggregate(r_i, t, mode)
        if len(r.shape) < 2:
            r = r.unsqueeze(0)

        mu, sigma = self.r_to_mu_sigma(r)
            
        return mu, sigma, h_w_prior

    def forward(self, h, prev_ts, ts, neg_samples=1, data=None):

        if self.base_model == "origin":
            if self.training:
                pos_pred, neg_pred = self.edge_predictor(
                    h, neg_samples=neg_samples)
                return pos_pred, neg_pred, None, None
            else:
                return self.edge_predictor(h, neg_samples=neg_samples)
        else:

            ts = ts / self.resize_ratio
            prev_ts = prev_ts / self.resize_ratio

            # TODO: verify diff between num_edge & bs

            num_edge = h.shape[0] // (neg_samples + 2)
            h_src = h[:num_edge]
            ts_src = ts[:num_edge]
            h_pos_dst = h[num_edge:2 * num_edge]
            ts_pos_dst = ts[num_edge:2 * num_edge]
            h_neg_dst = h[2 * num_edge:]
            ts_neg_dst = ts[2 * num_edge:]

            prev_ts_src = prev_ts[:num_edge]
            prev_ts_pos_dst = prev_ts[num_edge:2 * num_edge]
            prev_ts_neg_dst = prev_ts[2 * num_edge:]


            if self.training:
                if self.old_as_context:
                    if self.enable_ode and torch.max(ts) > self.last_ts:
                        if self.base_model == 'mnp':
                            t_0 = self.last_ts.expand(ts_src.shape[0], 1)
                            init_r = self.memory_net(ts_src.unsqueeze(-1))
                            context_r = self.ode_solver(init_r, t_0,
                                                        ts_src.unsqueeze(-1))
                        else:
                            context_r = self.ode_solver(self.running_r.unsqueeze(0), self.last_ts.view(1, 1),
                                                        torch.max(ts).view(1, 1))
                    else:
                        if self.base_model == "mnp":
                            context_r = self.memory_net(ts_src.unsqueeze(-1))
                        else:
                            context_r = self.running_r.unsqueeze(0)
                    mu_context, sigma_context = self.r_to_mu_sigma(context_r)

                else:
                    context_data, target_data, context_ts, target_ts = self.context_spliter(
                        [h_src, h_pos_dst, h_neg_dst, ts_src, ts_pos_dst, ts_neg_dst])
                    mu_context, sigma_context = self.xy_to_mu_sigma(context_data, context_ts,
                                                                    'context', negative_sample=neg_samples)

                mu_target, sigma_target, h_w_prior = self.xy_to_mu_sigma([h_src, h_pos_dst, h_neg_dst],
                                                              [prev_ts_src, prev_ts_pos_dst, prev_ts_neg_dst],
                                                              [ts_src, ts_pos_dst,
                                                               ts_neg_dst], 'target',
                                                              negative_sample=neg_samples)

                if self.base_model == 'mnp':
                    target_r = self.memory_net(ts_src.unsqueeze(-1))
                    mu_target, sigma_target = self.r_to_mu_sigma(target_r)
                if self.enabe_determinstic:
                    r = self.running_r.expand(h.shape[0], self.running_r.shape[0])
                    pos_pred, neg_pred, dist = self.deterministic_decoder(h, r, neg_samples=neg_samples, training=self.training)
                    return pos_pred, neg_pred, dist

                q_context = Normal(mu_context, sigma_context)
                q_target = Normal(mu_target, sigma_target)
                z_sample = q_target.rsample(torch.Size([self.l]))  # (l, z_dim)

                # target_pos_pred, target_neg_pred = self.decoder(
                #     h, z_sample, ts, self.last_ts, neg_samples=neg_samples)

                h__predict = self.decoder(
                    # h, z_sample, ts, self.last_ts, neg_samples=neg_samples
                    h_w_prior, z_sample, ts, self.last_ts, neg_samples=neg_samples
                )

                # mix_up to augment h
                # h__predict = h + h__predict
                h__predict = self.merge_aug(h, h__predict)
                # h__predict = h
                
                return h__predict, q_target, q_context

            else:
                if (data == "train" or data == 'val') and self.test:
                    mu_target, sigma_target, h_w_prior = self.xy_to_mu_sigma([h_src, h_pos_dst, h_neg_dst],
                                                                  [prev_ts_src, prev_ts_pos_dst, prev_ts_neg_dst],
                                                                  [ts_src, ts_pos_dst,
                                                                   ts_neg_dst], 'target',
                                                                  negative_sample=neg_samples)
                    if self.base_model == 'mnp':
                        target_r = self.memory_net(ts_src.unsqueeze(-1))
                        mu_target, sigma_target = self.r_to_mu_sigma(target_r)
                    if self.enabe_determinstic:
                        r = self.running_r.expand(h.shape[0], self.running_r.shape[0])
                        pos_pred, neg_pred = self.deterministic_decoder(h, r, neg_samples=neg_samples)
                        return pos_pred, neg_pred

                else:
                    if self.enable_ode and torch.max(ts) > self.last_ts:
                        if self.base_model == 'mnp':
                            init_r = self.memory_net(ts_src.unsqueeze(-1))
                            context_r = self.ode_solver(init_r, self.last_ts.expand(ts_src.shape[0]).unsqueeze(-1),
                                                        ts_src.unsqueeze(-1))
                        else:
                            r = self.running_r.expand(
                                ts_src.shape[0], self.running_r.shape[0])
                            t_0 = self.last_ts.expand(ts_src.shape[0])
                            context_r = self.ode_solver(
                                r, t_0.unsqueeze(-1), ts_src.unsqueeze(-1))
                    else:
                        if self.base_model == "mnp":
                            context_r = self.memory_net(ts_src.unsqueeze(-1))
                        else:
                            context_r = self.running_r.unsqueeze(0)
                    if self.enabe_determinstic:
                        r = context_r.expand(h.shape[0], context_r.shape[-1])
                        pos_pred, neg_pred = self.deterministic_decoder(h, r, neg_samples=neg_samples)
                        return pos_pred, neg_pred

                    mu_target, sigma_target = self.r_to_mu_sigma(context_r)

                q_target = Normal(mu_target, sigma_target)
                z_sample = q_target.rsample(torch.Size([self.l]))

                # target_pos_pred, target_neg_pred = self.decoder(
                #     h, z_sample, ts, self.last_ts, neg_samples=neg_samples)

                src_hid = h_src
                pos_dst_hid = h_pos_dst
                neg_dst_hid = h_neg_dst

                delta_ts_src_emb = self.time_encoder(ts_src-prev_ts_src)
                delta_ts_pos_dst_emb = self.time_encoder(ts_pos_dst-prev_ts_pos_dst)
                delta_ts_neg_dst_emb = self.time_encoder(ts_neg_dst-prev_ts_neg_dst)

                prev_ts_src_emb = self.time_encoder(prev_ts_src)
                prev_ts_pos_dst_emb = self.time_encoder(prev_ts_pos_dst)
                prev_ts_neg_dst_emb = self.time_encoder(prev_ts_neg_dst)

                ts_src_emb = self.time_encoder(ts_src)
                ts_pos_dst_emb = self.time_encoder(ts_pos_dst)
                ts_neg_dst_emb = self.time_encoder(ts_neg_dst)

                ts_src_fin = delta_ts_src_emb + prev_ts_src_emb + ts_src_emb
                ts_pos_dst_fin = delta_ts_pos_dst_emb + prev_ts_pos_dst_emb + ts_pos_dst_emb
                ts_neg_dst_fin = delta_ts_neg_dst_emb + prev_ts_neg_dst_emb + ts_neg_dst_emb

                src_w_prior = src_hid.view(-1,src_hid.shape[-1])
                pos_dst_w_prior = pos_dst_hid.view(-1,pos_dst_hid.shape[-1])
                neg_dst_w_prior = neg_dst_hid.view(-1,neg_dst_hid.shape[-1])

                h_w_prior = torch.cat([src_w_prior, pos_dst_w_prior, neg_dst_w_prior], dim=0)

                h__predict = self.decoder(
                    # h, z_sample, ts, self.last_ts, neg_samples=neg_samples
                    h_w_prior, z_sample, ts, self.last_ts, neg_samples=neg_samples
                )

                # mix_up to augment h
                # h__predict = h + h__predict
                h__predict = self.merge_aug(h, h__predict)
                # h__predict = h

                return h__predict

    def aggregate(self, r_i, ts, mode):

        t_emb = self.time_encoder(ts)
        r_i = r_i + t_emb
        current_r = torch.mean(r_i, dim=0)
        current_t = torch.max(ts)
        if self.base_model == "snp":
            r = self.update_cell(self.running_r, current_r)
            if mode == "target":
                self.running_r = r
                self.last_ts = current_t
        elif self.base_model == "np":
            if mode == "target":
                self.num_batches_tracked += 1
                momentum = 1 / float(self.num_batches_tracked)
                self.running_r = (1 - momentum) * \
                                 self.running_r + momentum * current_r
                self.last_ts = current_t
                r = self.running_r
            else:
                momentum = 1 / float(self.num_batches_tracked + 1)
                r = (1 - momentum) * self.running_r + momentum * current_r
        elif self.base_model == "anp":
            current_h = current_r.unsqueeze(0)
            memory = torch.cat([self.history_memory, current_h], dim=0)
            attn_out, _ = self.multi_atten(current_h, memory, memory, need_weights=False)
            r = torch.mean(attn_out, dim=0)
            if mode == "target":
                self.running_r = r
                self.history_memory = memory
                self.last_ts = current_t
        elif self.base_model == 'mnp':
            ts.unsqueeze_(-1)
            mem_out = self.memory_net.update(ts, r_i)
            r = self.memory_net(ts, mem_out)
            if mode == "target":
                self.memory_net.memory = mem_out
                self.last_ts = current_t
        return r


class MergeAUG(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(in_dim)
        # hidden_dim = (in_dim+out_dim)//2
        # self.fc1 = nn.Linear(in_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x1, x2):
        h = torch.cat([x1, x2], dim=1)
        h = self.norm(h)
        h = self.dropout(h)

        h = self.fc(h)
        
        return h 


class jump_attn_agger(nn.Module):
    def __init__(self, emb_dim, kv_dim, n_head=2, dropout=0.1):
        super(jump_attn_agger, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim,
                                                    kdim=kv_dim,
                                                    vdim=kv_dim,
                                                    num_heads=n_head,
                                                    dropout=dropout)

    def forward(self, left_mem, msg, delta_ts):

        mask = None
        attn_output, attn_weight = self.multihead_attn(left_mem.unsqueeze(0), msg.unsqueeze(0), msg.unsqueeze(0), mask)
        attn_output, attn_weight = attn_output.squeeze(), attn_weight.squeeze()

        return attn_output


class hShrinkTransform(nn.Module):

    def __init__(self, h_ori_dim, h_dim):
        super(hShrinkTransform, self).__init__()

        self.h_ori_dim = h_ori_dim
        self.h_dim = h_dim

        layers = [nn.Linear(h_ori_dim, h_dim)
                #   ,nn.LeakyReLU()
                ]

        self.h_to_hidden = nn.Sequential(*layers)

    def forward(self, h):
        
        return self.h_to_hidden(h)


class Encoder(nn.Module):

    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim
        self.y_emb = nn.Embedding(2, y_dim)
        layers = [
            # nn.Linear(x_dim + y_dim, h_dim),
            # nn.LeakyReLU(),
            # nn.Linear(h_dim, h_dim),
            # nn.LeakyReLU(),
            # nn.Linear(h_dim, r_dim)

            nn.Linear(x_dim + y_dim, r_dim)
        ]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):

        y = self.y_emb(y)
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, out_dim):
        super(Decoder, self).__init__()
        self.decode_fc = nn.Sequential(
            # nn.Linear(x_dim + z_dim, h_dim),
            # # nn.Linear(x_dim + z_dim, out_dim),
            # nn.LeakyReLU(),
            # nn.Linear(h_dim, out_dim)
            # # nn.Linear(out_dim, out_dim)

            nn.Linear(x_dim + z_dim, out_dim)
        )

    def forward(self, x, z, ts=None, t_0=None, neg_samples=1):

        z = z.transpose(0, 1)
        if z.shape[0] == 1:
            z = z.expand(x.shape[0], z.shape[1], z.shape[2])
        else:
            z = z.tile(neg_samples + 2, 1, 1)

        x = x.unsqueeze(1).expand(x.shape[0], z.shape[1], x.shape[1])
        x = torch.cat([x, z], dim=-1)

        h = self.decode_fc(x)
        h = torch.mean(h, dim=1)

        return h


class DeterminsticDecoder(nn.Module):

    def __init__(self, x_dim, r_dim, h_dim, out_dim):
        super(DeterminsticDecoder, self).__init__()
        self.decode_fc = nn.Sequential(nn.Linear(x_dim + r_dim, h_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(h_dim, out_dim))
        self.edge_predictor = EdgePredictor(out_dim, dim_out=2)

    def forward(self, x, r, ts=None, t_0=None, neg_samples=1, training=False):

        x = torch.cat([x, r], dim=-1)
        h = self.decode_fc(x)
        pos_pred, neg_pred = self.edge_predictor(h, neg_samples=neg_samples)
        pos_mean, pos_var = pos_pred.split(1, dim=-1)
        neg_mean, neg_var = neg_pred.split(1, dim=-1)
        mu = torch.cat([pos_mean, neg_mean])
        sigma = torch.cat([pos_var, neg_var])
        sigma = 0.1 + 0.9 * F.softplus(sigma)
        dist = Normal(mu, sigma)
        if training:
            return pos_mean, neg_mean, dist
        else:
            return pos_mean, neg_mean


class ODESolver(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(self, r_dim, h_dim, t_dim, time_enc, r_tol=1e-6, a_tol=1e-7, method="dopri5"):
        super(ODESolver, self).__init__()
        self.a_tol = a_tol
        self.r_tol = r_tol
        self.method = method
        self.ode_func = ODEFunc(r_dim, h_dim, t_dim,
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


class ODEDecoder(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(self, latent_odefunc, x_dim, z_dim, h_dim, resize_ratio, tol=1e-6, method="dopri5", ):
        super(ODEDecoder, self).__init__()
        self.tol = tol
        self.method = method

        self.resize_ratio = resize_ratio
        self.ode_func = ODEFunc(z_dim, h_dim, self.start_time, self.end_time)
        self.decode_fc = nn.Linear(x_dim + z_dim, h_dim)
        self.edge_predictor = EdgePredictor(h_dim)

    def forward(self, x, z, ts, t_0=None):

        z = z.expand(x.shape[0], z.shape[0], z.shape[1])
        x = x.unsqueeze(1).expand(
            x.shape[0], z.shape[1], x.shape[1])
        ts = ts.unsqueeze(-1)
        if t_0 is None:
            t_0 = torch.zeros_like(ts).to(ts)
        t_0 = t_0 / float(self.resize_ratio)
        ts = ts / float(self.resize_ratio)
        initial_state = (z, ts, t_0)
        tt = torch.tensor([self.start_time, self.end_time]).to(z)
        solution = odeint_adjoint(
            self.ode_func,
            initial_state,
            tt,
            method=self.method
        )

        z_final, _, _ = solution
        z_final = z_final[-1, :]
        x = torch.cat([x, z_final], dim=-1)
        b, l, dim = x.shape
        x = x.view(b * l, -1)
        h = self.decode_fc(x)
        pos_pred, neg_pred = self.edge_predictor(h)
        pos_pred = pos_pred.view(b // 3, l, -1)
        neg_pred = neg_pred.view(b // 3, l, -1)
        return torch.mean(pos_pred, dim=1), torch.mean(neg_pred, dim=1)

    def train_decoder(self, x, z):

        z = z.expand(x.shape[0], z.shape[0], z.shape[1])
        x = x.unsqueeze(1).expand(x.shape[0], z.shape[1], x.shape[1])
        x = torch.cat([x, z], dim=-1)
        b, l, dim = x.shape
        x = x.view(b * l, -1)
        h = self.decode_fc(x)
        pos_pred, neg_pred = self.edge_predictor(h)
        pos_pred = pos_pred.view(b // 3, l, -1)
        neg_pred = neg_pred.view(b // 3, l, -1)
        return torch.mean(pos_pred, dim=1), torch.mean(neg_pred, dim=1)


class ODEFunc(nn.Module):
    def __init__(self, z_dim, h_dim, t_dim, time_enc, start_time, end_time):
        super(ODEFunc, self).__init__()
        self.start_time = start_time
        self.end_time = end_time
        self.time_enc = time_enc
        ode_layers = [nn.Linear(z_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, z_dim)]
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


class EncODEFunc(nn.Module):
    def __init__(self, latent_odefunc):
        super(EncODEFunc, self).__init__()
        self.latent_odefunc = latent_odefunc

    def forward(self, t, x):
        t = t.view(1, 1)
        return self.latent_odefunc(torch.cat([x, t], dim=-1))


class MuSigmaEncoder(nn.Module):

    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):

        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class MemoryNet(nn.Module):
    def __init__(self, mem_size, h, ts_dim, inp_dim, time_enc, device):
        super(MemoryNet, self).__init__()
        self.mem_size = mem_size
        self.h = h
        self.inp_dim = inp_dim
        self.time_encoder = time_enc
        self.device = device
        self.memory = torch.zeros(mem_size, h, device=self.device)
        self.Q_linear = nn.Linear(ts_dim, h, bias=False)
        self.K_linear = nn.Linear(h, h, bias=False)
        self.V_linear = nn.Linear(inp_dim, h, bias=False)
        self.gru = nn.GRUCell(h, h)
        self.feat_droupout = nn.Dropout2d(0.2)
        self.reset()

    def update(self, ts, input):

        t_emb = self.time_encoder(ts)
        q_h = self.Q_linear(t_emb)
        k_h = self.K_linear(self.memory)
        weight = torch.softmax(torch.matmul(q_h, k_h.T),
                               dim=1)
        v_h = self.V_linear(input)
        new_mem = (weight.unsqueeze(2) * v_h.unsqueeze(1)).sum(0)
        mem = self.memory + new_mem
        return F.normalize(mem, p=2, dim=1)

    def forward(self, ts, mem=None):

        if mem is None:
            mem = self.memory
        t_emb = self.time_encoder(ts)
        q_h = self.Q_linear(t_emb)
        k_h = self.K_linear(mem)
        weight = torch.softmax(torch.matmul(q_h, k_h.T) / (k_h.shape[-1] ** (1 / 2)),
                               dim=1)
        read_r = torch.matmul(weight, mem)
        return read_r

    def reset(self):

        self.memory = torch.randn_like(self.memory, device=self.device)
