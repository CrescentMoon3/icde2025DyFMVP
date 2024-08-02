from typing import Optional, Tuple
import warnings
import math
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from .model.basic_modules import MLP
from .model.dyfmvp import DYFMVP
from .utils import BackgroundThreadGenerator

from sklearn.metrics import precision_recall_fscore_support, roc_curve

def compute_ks(label, preds):
    fpr, tpr, thresholds = roc_curve(label, preds)

    ks_value = max(tpr - fpr)

    return ks_value


def eval_edge_prediction(model: DYFMVP, dl: DataLoader,
                         device: torch.device,
                         restart_mode: bool,
                         uptodate_nodes: Optional[set]=None,
                         mean_over_n_samples: int=200):
    model.eval()
    it = BackgroundThreadGenerator(dl)
    val_ap = []
    val_auc = []
    pos_all_preds = []
    neg_all_preds = []
    uptodate_nodes = set() if uptodate_nodes is None else uptodate_nodes
    with torch.no_grad():
        it = tqdm.tqdm(it, total=len(dl), ncols=50)
        for src_ids, dst_ids, neg_dst_ids, ts, eids, _, comp_graph in it:
            src_ids = src_ids.long().to(device)
            dst_ids = dst_ids.long().to(device)
            neg_dst_ids = neg_dst_ids.long().to(device)
            ts = ts.float().to(device)
            eids = eids.long().to(device)
            comp_graph.to(device)

            if restart_mode:  # lazy restart
                involved_nodes = comp_graph.np_computation_graph_nodes
                restart_nodes = set(involved_nodes) - set(uptodate_nodes)
                r_nids = torch.tensor(list(restart_nodes)).long().to(device)
                model.restart(r_nids, torch.full((len(r_nids),), ts.min().item()).to(device))
                uptodate_nodes.update(restart_nodes)

            _1, _2, pos_scores, neg_scores, *_ = model.contrast_learning(
                src_ids, dst_ids, neg_dst_ids, ts, eids, comp_graph)

            pos_all_preds.append(pos_scores.sigmoid().cpu().numpy())
            neg_all_preds.append(neg_scores.sigmoid().cpu().numpy())

    pos_all_preds = np.concatenate(pos_all_preds)
    neg_all_preds = np.concatenate(neg_all_preds)
    n = math.ceil(len(pos_all_preds) / mean_over_n_samples)
    for i in range(n):
        l = i * mean_over_n_samples
        r = min((i+1) * mean_over_n_samples, len(pos_all_preds))
        bs = r - l
        pred_score = np.concatenate([pos_all_preds[l:r], neg_all_preds[l:r]])
        true_label = np.concatenate([np.ones(bs), np.zeros(bs)])
        valid_mask = np.isfinite(pred_score)
        # TODO: unreproducible bug: pred_score are all nan
        if not np.all(valid_mask):
            warnings.warn(f'Encounter invalid values: {pred_score[~valid_mask]}')
            pred_score = pred_score[valid_mask]
            true_label = true_label[valid_mask]

        val_ap.append(average_precision_score(true_label, pred_score))
        val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_ap), np.mean(val_auc)


@torch.no_grad()
def eval_node_classification(encoder: DYFMVP, decoder: MLP, dl: DataLoader,
                             device: torch.device
                             ) -> float:
    preds = []
    trues = []
    it = BackgroundThreadGenerator(dl)
    it = tqdm.tqdm(it, total=len(dl), ncols=50)
    encoder.eval()
    decoder.eval()
    for src_ids, dst_ids, neg_dst_ids, ts, eids, labels, comp_graph in it:
        bs = len(src_ids)
        src_ids = src_ids.long().to(device)
        dst_ids = dst_ids.long().to(device)
        neg_dst_ids = neg_dst_ids.long().to(device)
        ts = ts.float().to(device)
        eids = eids.long().to(device)
        # labels = labels.float().to(device)
        comp_graph.to(device)

        _, h, *_ = encoder.contrast_learning(src_ids, dst_ids, neg_dst_ids, 
                                             ts, eids, comp_graph)
        scores = decoder(h[:bs])
        preds.append(scores.sigmoid().cpu().numpy())
        trues.append(labels.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    eval_auc = roc_auc_score(trues, preds)
    return eval_auc

@torch.no_grad()
def eval_edge_classification(encoder: DYFMVP, decoder: MLP, dl: DataLoader,
                             device: torch.device, raw_feat_getter
                             ) -> float:
    preds = []
    trues = []
    it = BackgroundThreadGenerator(dl)
    it = tqdm.tqdm(it, total=len(dl), ncols=50)
    raw_feat_getter = raw_feat_getter
    encoder.eval()
    decoder.eval()
    for src_ids, dst_ids, neg_dst_ids, ts, eids, labels, comp_graph in it:
        bs = len(src_ids)
        src_ids = src_ids.long().to(device)
        dst_ids = dst_ids.long().to(device)
        neg_dst_ids = neg_dst_ids.long().to(device)
        ts = ts.float().to(device)
        eids = eids.long().to(device)
        comp_graph.to(device)

        _, h, *_ = encoder.contrast_learning(src_ids, dst_ids, neg_dst_ids, 
                                             ts, eids, comp_graph)
        
        batch_edge_feat = raw_feat_getter.get_edge_embeddings(eids.cpu()).to(device)
        
        scores = decoder(input_1=h[:bs], input_2=h[bs:], input_3=batch_edge_feat)
        
        
        preds.append(scores.sigmoid().cpu().numpy())
        trues.append(labels.numpy())
        
        # it.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')
        
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    eval_auc = roc_auc_score(trues, preds)
    return eval_auc


def warmup(model: DYFMVP, dl: DataLoader, device: torch.device,
           uptodate_nodes: Optional[set]=None):

    model.eval()
    it = BackgroundThreadGenerator(dl)
    uptodate_nodes = set() if uptodate_nodes is None else uptodate_nodes
    with torch.no_grad():
        it = tqdm.tqdm(it, total=len(dl), ncols=50)
        for src_ids, dst_ids, neg_dst_ids, ts, eids, _, comp_graph in it:
            src_ids = src_ids.long().to(device)
            dst_ids = dst_ids.long().to(device)
            neg_dst_ids = neg_dst_ids.long().to(device)
            ts = ts.float().to(device)
            eids = eids.long().to(device)
            comp_graph.to(device)

            # lazy restart
            involved_nodes = comp_graph.np_computation_graph_nodes
            restart_nodes = set(involved_nodes) - set(uptodate_nodes)
            r_nids = torch.tensor(list(restart_nodes)).long().to(device)
            model.restart(r_nids, torch.full((len(r_nids),), ts.min().item()).to(device))
            uptodate_nodes.update(restart_nodes)

            model.contrast_learning(src_ids, dst_ids, neg_dst_ids, ts, eids, comp_graph)

    return uptodate_nodes


@torch.no_grad()
def encode_trajectory(model: DYFMVP, dl: DataLoader, device: torch.device,
                      agg: str, use_src: bool=True, use_dst: bool=True):
    node_reprs = np.zeros((model.n_nodes, model.nfeat_dim))
    node_counts = np.zeros(model.n_nodes)
    
    model.eval()
    model.reset()
    it = BackgroundThreadGenerator(dl)
    it = tqdm.tqdm(it, total=len(dl), ncols=50)
    for src_ids, dst_ids, neg_dst_ids, ts, eids, labels, comp_graph in it:
        bs = len(src_ids)
        np_src_ids = src_ids.numpy()
        np_dst_ids = dst_ids.numpy()

        src_ids = src_ids.long().to(device)
        dst_ids = dst_ids.long().to(device)
        neg_dst_ids = neg_dst_ids.long().to(device)
        ts = ts.float().to(device)
        eids = eids.long().to(device)
        labels = labels.long().to(device)
        comp_graph.to(device)

        _, h,  *_ = model.contrast_learning(
            src_ids, dst_ids, neg_dst_ids, ts, eids, comp_graph
        )
        h = h.cpu().numpy()
        
        if use_src:
            for i, node in enumerate(np_src_ids):
                if agg == 'last':
                    node_reprs[node] = h[i]
                elif agg == 'max':
                    node_reprs[node] = np.maximum(node_reprs[node], h[i])
                else:
                    node_reprs[node] += h[i]
                node_counts[node] += 1
        if use_dst:
            for j, node in enumerate(np_dst_ids):
                i = j + bs
                if agg == 'last':
                    node_reprs[node] = h[i]
                elif agg == 'max':
                    node_reprs[node] = np.maximum(node_reprs[node], h[i])
                else:
                    node_reprs[node] = h[i]
                node_counts[node] += 1
        
    if agg == 'mean':
        node_reprs /= (node_counts[:, None] + 1e-7)
        
    return node_reprs
