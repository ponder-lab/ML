# `tf.scatter_nd(indices, updates, shape)` takes its dtype from `updates` and its shape from
# `shape`; the graph builders below mirror an edge-count builder: float32 ones per edge scattered
# onto a per-node vector whose length is the node count.
import tensorflow as tf


def consume_counts(c):
    return c


def consume_counts_static(s):
    return s


def consume_gathered(g):
    return g


def incoming_edges_num(node_embeddings, targets):
    indices = tf.expand_dims(targets, -1)
    return tf.scatter_nd(
        indices=indices,
        updates=tf.ones_like(targets, dtype=tf.float32),
        shape=(tf.shape(node_embeddings)[0],),
    )


node_embeddings = tf.ones((5, 3))
targets = tf.constant([0, 2, 2, 4])
counts = incoming_edges_num(node_embeddings, targets)
assert counts.shape == (5,) and counts.dtype == tf.float32
consume_counts(counts)

static = tf.scatter_nd(tf.constant([[1], [3]]), tf.constant([7, 9]), shape=[6])
assert static.shape == (6,) and static.dtype == tf.int32
consume_counts_static(static)

gathered = tf.gather(counts, targets)
assert gathered.shape == (4,) and gathered.dtype == tf.float32
consume_gathered(gathered)
