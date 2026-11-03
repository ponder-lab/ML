# Test https://github.com/wala/ML/issues/164: `map` calls its function with a tuple element's
# members as separate arguments and any other element whole, and its element is what the function
# returns, so each member keeps its own type.
import tensorflow as tf


def consume_param_x(x):
    pass


def consume_param_ids(ids):
    pass


def consume_param_row(row):
    pass


def consume_param_keyed(d):
    pass


def consume_pair(p):
    pass


def consume_x(x):
    pass


def consume_ids(ids):
    pass


def consume_row(row):
    pass


def consume_keyed_a(a):
    pass


def consume_batched_x(x):
    pass


pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.zeros([8, 3], tf.int32))
)
rows = tf.data.Dataset.from_tensor_slices(tf.ones([8, 5]))
keyed = tf.data.Dataset.from_tensor_slices(
    {"a": tf.ones([8, 2]), "b": tf.zeros([8], tf.int32)}
)


def scale_pair(x, ids):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_param_x(x)
    assert ids.shape == (3,) and ids.dtype == tf.int32
    consume_param_ids(ids)
    return x * 2.0, ids


def scale_row(row):
    assert row.shape == (5,) and row.dtype == tf.float32
    consume_param_row(row)
    return row * 2.0


def pick_a(d):
    assert isinstance(d, dict)
    consume_param_keyed(d)
    return d["a"]


for p in pairs.map(scale_pair):
    assert isinstance(p, tuple) and len(p) == 2
    consume_pair(p)

for x, ids in pairs.map(scale_pair):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_x(x)
    assert ids.shape == (3,) and ids.dtype == tf.int32
    consume_ids(ids)

for row in rows.map(scale_row):
    assert row.shape == (5,) and row.dtype == tf.float32
    consume_row(row)

for a in keyed.map(pick_a):
    assert a.shape == (2,) and a.dtype == tf.float32
    consume_keyed_a(a)

for x, ids in pairs.map(scale_pair).batch(2):
    assert x.shape == (2, 4) and x.dtype == tf.float32
    consume_batched_x(x)
