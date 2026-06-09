from typing import NamedTuple, List

import tensorflow as tf


# `GNNInput` is the only symbol reachable from `GCNLayer.call` in this package;
# the rest of upstream `nlpgnn/gnn/utils.py` (scipy-based graph preprocessing) is
# off the analyzed path and omitted. The definition below is verbatim from
# upstream `kyzhouhzau/NLPGNN/nlpgnn/gnn/utils.py`.
class GNNInput(NamedTuple):
    node_embeddings: tf.Tensor
    adjacency_lists: List
