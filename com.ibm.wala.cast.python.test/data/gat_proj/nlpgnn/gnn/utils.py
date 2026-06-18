#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""

# Reachable slice of `kyzhouhzau/NLPGNN/nlpgnn/gnn/utils.py` for `GATLayer.call`:
# `GNNInput` (the message-passing input tuple), plus `masksoftmax` and its helper
# `maybe_num_nodes`, which `GraphAttentionConvolution.message_function` calls. The
# upstream module additionally defines self-loop, coalesce, and sparse helpers that
# are not reachable from this driver, so they are omitted to keep the analyzed slice
# minimal (the `GNNInput`/`MessagePassing`/`GraphAttentionConvolution` modules are
# vendored verbatim).
from typing import NamedTuple, List
import tensorflow as tf


class GNNInput(NamedTuple):
    node_embeddings: tf.Tensor
    adjacency_lists: List


def maybe_num_nodes(index, num_nodes):
    return tf.reduce_max(index) + 1 if num_nodes is None else num_nodes


def masksoftmax(src, index, num_nodes=None):
    num_nodes = maybe_num_nodes(index, num_nodes)
    inter = tf.math.unsorted_segment_max(
        data=src, segment_ids=index, num_segments=num_nodes
    )
    # out = src - tf.gather(inter, index)# 每一个维度减去最大的特征
    out = src
    out = tf.math.exp(out)
    inter = tf.math.unsorted_segment_sum(
        data=out, segment_ids=index, num_segments=num_nodes
    )
    out = out / (tf.gather(inter, index) + 1e-16)
    return out
