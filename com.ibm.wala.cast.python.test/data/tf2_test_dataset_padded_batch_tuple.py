# Test wala/ML#759: batch transforms must distribute into tuple-structured dataset
# elements. The chain mirrors the vendored gpt-2 `input_fn`: a `map` producing a
# tuple element, then `padded_batch`, `repeat`, and `prefetch`, iterated with the
# nested-unpack `enumerate` pattern.
import tensorflow as tf


def to_pair(x):
    return x, x


def consume_first(t):
    assert t.shape == (4, 3)
    assert t.dtype == tf.int32


def consume_second(t):
    assert t.shape == (4, 3)
    assert t.dtype == tf.int32


dataset = tf.data.Dataset.from_tensor_slices(tf.ones((8, 3), dtype=tf.int32))
dataset = dataset.map(to_pair)
dataset = dataset.padded_batch(4, padded_shapes=([-1], [-1]))
dataset = dataset.repeat(2)
dataset = dataset.prefetch(1)

for step, (a, b) in enumerate(dataset):
    consume_first(a)
    consume_second(b)
