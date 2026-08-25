import tensorflow as tf


def consume_direct_indices(i):
    pass


def consume_direct(g):
    pass


def consume_listed_indices(i):
    pass


def consume_listed(g):
    pass


# Tier 1: the indices are a column sliced out of a parameter. `tf.gather`'s result is indexed
# by the indices' shape with each entry a row of the table, so the trailing axis is the
# table's regardless of what the indices resolve to.
def propagate_direct(node_embeddings, adjacency):
    edge_sources = adjacency[:, 0]
    assert edge_sources.shape == (30,)
    assert edge_sources.dtype == tf.int32
    consume_direct_indices(edge_sources)

    edge_source_states = tf.gather(node_embeddings, edge_sources)
    assert edge_source_states.shape == (30, 16)
    assert edge_source_states.dtype == tf.float32
    consume_direct(edge_source_states)


# Tier 2: the same, one container hop further out, which is the shape the message-passing
# idiom actually takes: the adjacency lists arrive as a sequence and are enumerated before a
# column is sliced from each.
def propagate_listed(node_embeddings, adjacency_lists):
    for edge_type_idx, adjacency_list in enumerate(adjacency_lists):
        edge_sources = adjacency_list[:, 0]
        assert edge_sources.shape == (30,)
        assert edge_sources.dtype == tf.int32
        consume_listed_indices(edge_sources)

        edge_source_states = tf.gather(node_embeddings, edge_sources)
        assert edge_source_states.shape == (30, 16)
        assert edge_source_states.dtype == tf.float32
        consume_listed(edge_source_states)


node_embeddings = tf.ones((100, 16), dtype=tf.float32)
adjacency = tf.ones((30, 2), dtype=tf.int32)

propagate_direct(node_embeddings, adjacency)
propagate_listed(node_embeddings, [adjacency])


def consume_variable_bounds(v):
    pass


# A slice whose bounds are variables rather than literals. The construction is the same
# `slice(lower, upper, step)`; only the bounds differ, and the axis still reduces.
def propagate_variable_bounds(adjacency, lo, hi):
    window = adjacency[lo:hi, 0]
    assert window.shape == (10,)
    consume_variable_bounds(window)


propagate_variable_bounds(adjacency, 0, 10)
