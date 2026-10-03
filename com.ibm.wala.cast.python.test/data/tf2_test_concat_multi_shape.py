# A concat whose known element has two possible shapes, beside an element whose shape is unknown
# (wala/ML#1009): the attention cache's `tf.concat([past_key, key], axis=-2)`, where `key` is typed
# differently in the contexts that reach it and the cached `past_key` is a value the analysis
# cannot shape. `tf.concat` requires every element to have the same rank and the same non-axis
# extents, so each of `key`'s shapes fixes the result's rank and its non-axis extents, and only the
# axis extent stays open.
import numpy as np
import tensorflow as tf


def consume(merged):
    pass


def make_key(single):
    if single:
        return tf.ones((1, 3, 4))
    return tf.ones((2, 3, 4))


def attend(key, past):
    merged = tf.concat([past, key], axis=1)
    assert merged.shape[0] == key.shape[0]
    assert merged.shape[1:] == (5, 4)
    assert merged.dtype == tf.float32
    consume(merged)


def run(single):
    key = make_key(single)
    batch = 1 if single else 2
    cached = np.frombuffer(
        np.zeros(batch * 2 * 4, dtype=np.float32).tobytes(), dtype=np.float32
    )
    past = tf.cast(cached.reshape(batch, 2, 4), tf.float32)
    attend(key, past)


run(True)
run(False)
