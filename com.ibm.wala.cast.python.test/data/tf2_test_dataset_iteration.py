# Test https://github.com/wala/ML/issues/1010: iterating a dataset yields its elements, not the
# dataset itself. A loop variable, its unpacked components, a dict-keyed component, and the elements
# that enumerate and zip pass through all carry the element's shape and dtype.
import sys

import numpy as np
import tensorflow as tf


def consume_x(t):
    assert t.shape == (8, 4)
    assert t.dtype == tf.float32
    return t


def consume_y(t):
    assert t.shape == (8,)
    assert t.dtype == tf.int32
    return t


def consume_single(t):
    assert t.shape == (8, 4)
    assert t.dtype == tf.float32
    return t


def consume_keyed(t):
    assert t.shape == (8, 4)
    assert t.dtype == tf.float32
    return t


def consume_enumerated(t):
    assert t.shape == (8, 4)
    assert t.dtype == tf.float32
    return t


def consume_zipped(t):
    assert t.shape == (8, 4)
    assert t.dtype == tf.float32
    return t


def consume_next(t):
    assert t.shape == (8, 4)
    assert t.dtype == tf.float32
    return t


def consume_row(t):
    assert t.shape == (4,)
    assert t.dtype == tf.float32
    return t


def consume_either(t):
    assert t.shape in ((8, 4), (8, 3))
    assert t.dtype == tf.float32
    return t


def consume_either_x(t):
    assert t.shape in ((8, 4), (8, 5))
    assert t.dtype == tf.float32
    return t


features = np.ones((16, 4), dtype=np.float32)
labels = np.zeros((16,), dtype=np.int32)

pairs = tf.data.Dataset.from_tensor_slices((features, labels)).batch(8)
for x, y in pairs:
    consume_x(x)
    consume_y(y)

singles = tf.data.Dataset.from_tensor_slices(features).batch(8)
for batch in singles:
    consume_single(batch)

keyed = tf.data.Dataset.from_tensor_slices({"a": features}).batch(8)
for record in keyed:
    consume_keyed(record["a"])

for step, (x2, y2) in enumerate(pairs):
    consume_enumerated(x2)

for (x3, y3), batch3 in zip(pairs, singles):
    consume_zipped(x3)
    consume_zipped(batch3)

consume_next(next(iter(singles)))

for batch2 in singles:
    consume_row(batch2[0])

narrow = tf.data.Dataset.from_tensor_slices(np.ones((16, 3), dtype=np.float32)).batch(8)
wide = tf.data.Dataset.from_tensor_slices(
    (np.ones((16, 5), dtype=np.float32), labels)
).batch(8)
flag = len(sys.argv) > 5
for either in singles if flag else narrow:
    consume_either(either)

for xe, ye in pairs if flag else wide:
    consume_either_x(xe)
