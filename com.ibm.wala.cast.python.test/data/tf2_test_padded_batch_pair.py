# Probe for wala/ML#688: a map stage returning a tuple, batched with a tuple `padded_shapes`,
# iterated with destructuring — the vendored gpt-2 `input_fn` element shape in miniature.
import tensorflow as tf


def add_labels(x):
    return x, x + 1


def consume(t):
    pass


def consume2(t):
    pass


dataset = tf.data.Dataset.from_tensor_slices(tf.ones((8, 3), dtype=tf.int64))
dataset = dataset.map(add_labels)
dataset = dataset.padded_batch(4, padded_shapes=([-1], [-1]))

for x, y in dataset:
    consume(x)
    consume2(y)

    assert x.shape == (4, 3)
    assert y.shape == (4, 3)
    assert x.dtype == tf.int64
    assert y.dtype == tf.int64
    break
