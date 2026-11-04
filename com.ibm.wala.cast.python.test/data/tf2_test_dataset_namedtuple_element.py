# Test https://github.com/wala/ML/issues/1210: a dataset element that is a namedtuple keeps its
# members as objects of their own, read by name as by position.
import collections

import tensorflow as tf

P = collections.namedtuple("P", ["x", "m"])


def consume_named_x(x):
    pass


def consume_named_m(m):
    pass


def consume_unpacked_m(m):
    pass


def consume_batched_m(m):
    pass


def consume_keyword_m(m):
    pass


def consume_whole_m(m):
    pass


pairs = tf.data.Dataset.from_tensor_slices(P(tf.ones([8, 2]), tf.zeros([8], tf.int32)))
for p in pairs:
    assert p.x.shape == (2,) and p.x.dtype == tf.float32
    consume_named_x(p.x)
    assert p.m.shape == () and p.m.dtype == tf.int32
    consume_named_m(p.m)
    x, m = p
    assert m.shape == () and m.dtype == tf.int32
    consume_unpacked_m(m)

for p in pairs.batch(2):
    assert p.m.shape == (2,) and p.m.dtype == tf.int32
    consume_batched_m(p.m)

keyed = tf.data.Dataset.from_tensor_slices(
    P(m=tf.zeros([8], tf.int32), x=tf.ones([8, 2]))
)
for p in keyed:
    assert p.m.shape == () and p.m.dtype == tf.int32
    consume_keyword_m(p.m)

# A dataset of one element that is a namedtuple, read by name.
for p in tf.data.Dataset.from_tensors(P(tf.ones([2]), tf.zeros([], tf.int32))):
    assert p.m.shape == () and p.m.dtype == tf.int32
    consume_whole_m(p.m)
