# Probe for the collection-dataflow family (wala/ML#570/#659): a column slice of a tensor taken
# from a list element keeps a tensor type (mirroring `adjacency_type_list[:, 1]` over
# `adjacency_lists`).
import tensorflow as tf


def consume(t):
    pass


adjacency_lists = [tf.constant([[0, 1], [1, 2], [2, 3], [3, 0]])]

for adjacency_type_list in adjacency_lists:
    targets = adjacency_type_list[:, 1]
    consume(targets)
    assert targets.shape == (4,)
    assert targets.dtype == tf.int32
