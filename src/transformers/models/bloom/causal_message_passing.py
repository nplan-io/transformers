""" A set of functions to perform message passing on a serialized graph in an LLM """
import enum
from collections import defaultdict
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch_geometric

from .desequence_graph_ids import SequenceElement


class GNNLayerFactory(enum.Enum):
    gcn = torch_geometric.nn.GCNConv
    sage = torch_geometric.nn.SAGEConv
    gat = torch_geometric.nn.GATConv


def build_message_passing_matrices(
    token_ids: torch.Tensor,
    edge_sequences: List[List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]]
) -> List[Dict[str, torch.Tensor]]:
    """ Returns the adjacency matrices required to perform causal message passing in between
        language model blocks of an autoregressive language model
    """
    message_passing_dicts = []
    for edge_sequence in edge_sequences:
        message_passing_dict = {'tokens2edges': [], 'edges2tokens': [], 'inverse_edge_index': []}
        node2edge_idxs = defaultdict(list)
        assert len(edge_sequence) > 0
        for edge_idx, sequenced_edge in enumerate(edge_sequence):
            pred_node, edge, succ_node = sequenced_edge
            node2edge_idxs[pred_node.ids].append(edge_idx)
            if isinstance(succ_node, SequenceElement):
                end_idx = succ_node.end_idx
                node2edge_idxs[succ_node.ids].append(edge_idx)
            elif isinstance(edge, SequenceElement):
                end_idx = edge.end_idx
            else:
                end_idx = pred_node.end_idx
            message_passing_dict['tokens2edges'].append(end_idx - 1)
            message_passing_dict['edges2tokens'].append(pred_node.start_idx)
        message_passing_dict['tokens2edges'] = torch.from_numpy(
            np.array(message_passing_dict['tokens2edges'])
        ).long().to(token_ids.device)
        message_passing_dict['edges2tokens'] = torch.from_numpy(
            np.array(message_passing_dict['edges2tokens'])
        ).long().to(token_ids.device)
        message_passing_dict['edge_index'] = []
        for edge_idxs in node2edge_idxs.values():
            if len(edge_idxs) < 2:
                continue
            for (idx0, idx1) in itertools.combinations(list(set(edge_idxs)), 2):
                message_passing_dict['edge_index'].append(
                    [idx0, idx1] if idx0 < idx1 else [idx1, idx0]
                )
        if len(message_passing_dict['edge_index']) > 0:
            message_passing_dict['edge_index'] = torch.from_numpy(
                np.array(message_passing_dict['edge_index']).transpose(1, 0)
            ).long().to(token_ids.device)
        else:
            message_passing_dict['edge_index'] = torch.from_numpy(
                np.array(message_passing_dict['edge_index'])
            ).long().to(token_ids.device)
        message_passing_dicts.append(dict(message_passing_dict))
    return message_passing_dicts


class CausalMessagePassingLayer(torch.nn.Module):
    def __init__(self, gnn_type: str, embedding_size: int):
        super().__init__()
        self.gnn_layer = GNNLayerFactory[gnn_type].value(embedding_size, embedding_size)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        message_passing_dicts: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        new_token_embeddings = []
        for t_embeddings, message_passing_dict in zip(token_embeddings, message_passing_dicts):
            edge_embeddings = t_embeddings[message_passing_dict['tokens2edges']]
            if message_passing_dict['edge_index'].numel() > 0:
                edge_embeddings = self.gnn_layer(
                    edge_embeddings,
                    message_passing_dict['edge_index']
                )
            if edge_embeddings.shape[0] > 1:
                causal_edge_embeddings = torch.cat([
                        torch.zeros_like(edge_embeddings[0]).unsqueeze(0),
                        edge_embeddings[:-1],
                ], dim=0)
            else:
                causal_edge_embeddings = torch.zeros_like(edge_embeddings)
            new_t_embeddings = torch.zeros_like(t_embeddings)
            new_t_embeddings[message_passing_dict['edges2tokens']] = causal_edge_embeddings
            new_token_embeddings.append(new_t_embeddings.unsqueeze(0))
        return token_embeddings + torch.cat(new_token_embeddings, dim=0)

# def perform_causal_message_passing(
#     token_embeddings: torch.Tensor,
#     message_passing_dicts: List[Dict[str, torch.Tensor]],
#     gnn_layer: Callable,
#     linear_layer: Callable,
#     reduce: str = 'mean'
# ) -> torch.Tensor:
#     """ Returns token embeddings in a sequence where causal message passing has been performed on
#         the token ids  based on the serialized graph described in the sequence
#     """
#     new_token_embeddings = []
#     for t_embeddings, message_passing_dict in zip(token_embeddings, message_passing_dicts):
#         edge_embeddings = scatter(
#             src=t_embeddings[message_passing_dict['tokens2edges'][0]],
#             dim=0,
#             index=message_passing_dict['tokens2edges'][1],
#             reduce=reduce
#         )
#         if message_passing_dict['edge_index'].numel() > 0:
#             edge_embeddings = gnn_layer(edge_embeddings, message_passing_dict['edge_index'])
#         if edge_embeddings.shape[0] > 1:
#             causal_edge_embeddings = torch.cat([
#                     torch.zeros_like(edge_embeddings[0]).unsqueeze(0),
#                     edge_embeddings[:-1],
#                     torch.zeros_like(edge_embeddings[0]).unsqueeze(0)
#             ], dim=0)
#         else:
#             causal_edge_embeddings = torch.cat([
#                     torch.zeros_like(edge_embeddings[0]).unsqueeze(0),
#                     torch.zeros_like(edge_embeddings[0]).unsqueeze(0)
#             ], dim=0)
#         new_t_embeddings = scatter(
#             src=causal_edge_embeddings[message_passing_dict['edges2tokens'][0]],
#             dim=0,
#             index=message_passing_dict['edges2tokens'][1],
#             reduce=reduce
#         )
#         assert new_t_embeddings.shape == t_embeddings.shape
#         new_token_embeddings.append(new_t_embeddings.unsqueeze(0))
#     return linear_layer(torch.cat([token_embeddings, torch.cat(new_token_embeddings, dim=0)], dim=2))
