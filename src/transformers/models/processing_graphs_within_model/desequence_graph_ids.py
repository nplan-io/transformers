""" A set of functions to identify a serialized graph within a list of token ids """

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SequenceElement:
    """ A data class for representing an element in a sequence which is an element of a serialized
        graph
    """
    token: int
    start_idx: int
    end_idx: int
    ids: Tuple[int]
    length: int


def add_descriptor_tokens(
    token_ids: torch.Tensor,
    graph_tokens: Dict[str, List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Adds descriptor tokens to token sequences after each fully described edge except the last
    """
    max_length = 0
    new_token_ids = []
    for t_ids in token_ids:
        sequence = _extract_graph_elements(t_ids.tolist(), graph_tokens)
        if len(sequence) == 0:
            new_token_ids.append(t_ids.tolist())
            continue
        else:
            new_t_ids = t_ids.tolist()[:sequence[0].start_idx] if sequence[0].start_idx > 0 else []
        pred_node_found = False
        for sequence_element in sequence:
            if not pred_node_found and sequence_element.token == graph_tokens['pred_node']:
                pred_node_found = True
            elif pred_node_found and sequence_element.token == graph_tokens['pred_node']:
                new_t_ids.append(graph_tokens['descriptor'])
            new_t_ids.extend([sequence_element.token] + list(sequence_element.ids))
        if sequence[-1].end_idx != len(t_ids):
            new_t_ids.extend(t_ids.tolist()[sequence[-1].end_idx:])
        max_length = max(max_length, len(new_t_ids))
        new_token_ids.append(new_t_ids)
    for batch_idx, t_ids in enumerate(new_token_ids):
        additional_length = max_length - len(t_ids)
        padding = additional_length * [graph_tokens['pad']]
        if additional_length > 0:
            new_token_ids[batch_idx].extend(padding)
        new_token_ids[batch_idx] = torch.from_numpy(
            np.array(new_token_ids[batch_idx])
        ).long().to(token_ids.device).unsqueeze(0)
    new_token_ids = torch.cat(new_token_ids, dim=0)
    return new_token_ids, torch.where(new_token_ids == graph_tokens['pad'], 0, 1).bool()


def remove_descriptor_tokens(
    token_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    descriptor_id: int
) -> torch.Tensor:
    """ Removes the embedding of descriptor tokens from token embedding sequences """
    new_token_embeddings = []
    min_length = np.inf
    for t_ids, t_embeds in zip(token_ids, token_embeddings):
        idxs2keep = [idx for idx, id in enumerate(t_ids.tolist()) if id != descriptor_id]
        new_token_embeddings.append(t_embeds[idxs2keep])
        min_length = min(min_length, new_token_embeddings[-1].shape[0])
    for batch_idx, new_embeds in enumerate(new_token_embeddings):
        new_token_embeddings[batch_idx] = new_embeds[:min_length].unsqueeze(0)
    return torch.cat(new_token_embeddings, dim=0)


def extract_edge_sequence(
    token_ids: List[int],
    graph_tokens: Dict[str, List[int]]
) -> List[Tuple[SequenceElement, Optional[SequenceElement], Optional[SequenceElement]]]:
    """ Returns a list of edges of the graph sequence identified in a sequence of generated token ids. """
    sequence = _extract_graph_elements(token_ids, graph_tokens)
    edges = []
    if len(sequence) > 2:
        for elem0, elem1, elem2 in zip(sequence[:-2], sequence[1:-1], sequence[2:]):
            if (
                elem0.token == graph_tokens['pred_node']
                and elem1.token == graph_tokens['edge']
                and elem2.token == graph_tokens['succ_node']
            ): # edge syntax
                edges.append((elem0, elem1, elem2))
    if (
        len(sequence) > 1
        and sequence[-2].token == graph_tokens['pred_node']
        and sequence[-1].token == graph_tokens['edge']
    ):
        edges.append((sequence[-2], sequence[-1], None))
    elif len(sequence) > 0 and sequence[-1].token == graph_tokens['pred_node']:
        edges.append((sequence[-1], None, None))
    return edges


def _extract_graph_elements(
    token_ids: List[int],
    graph_tokens: Dict[str, List[int]]
) -> List[SequenceElement]:
    """ Returns a parsable representation of the serialized graph in a sequence of token ids,
        if none is found, returns an empty list
    """
    sequence = []
    prev_token_id, prev_idx, final_idx = None, -1, len(token_ids)
    for token_idx, token_id in enumerate(token_ids):
        if token_id == graph_tokens['pred_node'] and prev_token_id is None:
            prev_token_id, prev_idx = token_id, token_idx
        elif (
            token_id in [graph_tokens['pred_node'], graph_tokens['edge'], graph_tokens['succ_node']]
            and prev_token_id is not None
        ):
            sequence.append(SequenceElement(
                token=prev_token_id,
                start_idx=prev_idx,
                end_idx=token_idx,
                ids=tuple(token_ids[prev_idx:token_idx])[1:],
                length=token_idx - prev_idx
            ))
            prev_token_id, prev_idx = token_id, token_idx
        elif token_id in [graph_tokens['eos'], graph_tokens['pad']] and prev_token_id is not None:
            final_idx = token_idx
            break
    if prev_token_id is not None:
        sequence.append(SequenceElement(
            token=prev_token_id,
            start_idx=prev_idx,
            end_idx=final_idx,
            ids=tuple(token_ids[prev_idx:final_idx])[1:],
            length=final_idx - prev_idx
        ))
    return sequence
