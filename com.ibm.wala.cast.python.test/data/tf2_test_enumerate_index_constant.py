# Test https://github.com/wala/ML/issues/1246: an `enumerate` index over a list of known length,
# and `len` of one, has the values the list's length fixes, until the list may grow.
import tensorflow as tf


def consume_from_parameter(x):
    assert x.shape == (2,)
    assert x.dtype == tf.float32


def consume_from_literal(x):
    assert x.shape == (2,)
    assert x.dtype == tf.float32


def consume_from_len(x):
    assert x.shape == (3,)
    assert x.dtype == tf.float32


def consume_from_returned_len(x):
    assert x.shape == (3,)
    assert x.dtype == tf.float32


def consume_from_constant(x):
    assert x.shape == (2,)
    assert x.dtype == tf.float32


def consume_from_appended(x):
    assert x.shape == (2,) or x.shape == (3,)
    assert x.dtype == tf.float32


def consume_from_extended(x):
    assert x.shape == (2,) or x.shape == (3,)
    assert x.dtype == tf.float32


def consume_from_appended_len(x):
    assert x.shape == (2,)
    assert x.dtype == tf.float32


def message(states, edge_type_idx):
    return states * (edge_type_idx + 1)


def propagate(states, adjacency_lists):
    for edge_type_idx, adjacency in enumerate(adjacency_lists):
        consume_from_parameter(tf.ones([edge_type_idx + 2]))
        message(states, edge_type_idx)


def load(features):
    return len(features)


edges = tf.constant([[0], [1]])
propagate(tf.ones((3, 4)), [edges])

for index, item in enumerate([edges]):
    consume_from_literal(tf.ones([index + 2]))

size = len([1.0, 2.0, 3.0])
consume_from_len(tf.ones([size]))

consume_from_returned_len(tf.ones([load([1.0, 2.0, 3.0])]))

zero = 0
consume_from_constant(tf.ones([zero + 2]))

appended = [edges]
appended.append(edges)
for position, item in enumerate(appended):
    consume_from_appended(tf.ones([position + 2]))

extended = [edges]
extended.extend([edges])
for position, item in enumerate(extended):
    consume_from_extended(tf.ones([position + 2]))

grown = [1.0]
grown.append(2.0)
consume_from_appended_len(tf.ones([len(grown)]))


def consume_from_started(x):
    assert x.shape == (3,)
    assert x.dtype == tf.float32


def consume_from_started_keyword(x):
    assert x.shape == (4,)
    assert x.dtype == tf.float32


# An index from a `start` does not count from `0`.
for position, item in enumerate([edges], 1):
    consume_from_started(tf.ones([position + 2]))

for position, item in enumerate([edges], start=2):
    consume_from_started_keyword(tf.ones([position + 2]))


def consume_from_started_double_starred(x):
    assert x.shape == (5,)
    assert x.dtype == tf.float32


for position, item in enumerate([edges], **{"start": 3}):
    consume_from_started_double_starred(tf.ones([position + 2]))
