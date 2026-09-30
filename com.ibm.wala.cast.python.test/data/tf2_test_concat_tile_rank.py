# Test https://github.com/wala/ML/issues/985: an edge list extended with self loops keeps its rank,
# so the edge indices and the gathered node states read through it keep theirs. `tf.tile` keeps its
# input's rank whatever its extents, and `tf.concat` takes its rank from any element whose shape is
# known, since every element must have the same one.
import sys

import tensorflow as tf


def consume_loops(x):
    pass


def consume_edges(x):
    pass


def consume_src(x):
    pass


def consume_index(x):
    pass


def consume_mismatch(x):
    pass


def add_remain_self_loop(adjacency_lists, num_nodes):
    loop_index = tf.range(0, num_nodes)
    loop_index = tf.expand_dims(loop_index, 1)
    loop_index = tf.tile(loop_index, [1, 2])
    consume_loops(loop_index)
    row = adjacency_lists[:, 0]
    col = adjacency_lists[:, 1]
    mask = row != col
    loop_index = tf.concat([adjacency_lists[mask], loop_index], 0)
    return loop_index


def masksoftmax(src, index):
    consume_src(src)
    consume_index(index)
    return src


num_nodes = len(sys.argv) + 2
edges = add_remain_self_loop(
    tf.constant([[0, 1], [1, 2], [2, 2]], dtype=tf.int32), num_nodes
)
assert edges.shape == (5, 2)
consume_edges(edges)
node_embeddings = tf.ones((num_nodes, 8))
targets = edges[:, 1]
states = tf.gather(node_embeddings, targets)
assert states.shape == (5, 8) and targets.shape == (5,)
masksoftmax(states, targets)

# Elements of known but different ranks cannot be concatenated, so the result stays unknown.
try:
    consume_mismatch(tf.concat([tf.ones((2, 2)), tf.ones((2, 2, 2))], 0))
except Exception:
    pass
