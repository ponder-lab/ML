# Test `tf.random.categorical`, which draws `num_samples` class indices per row of a `(batch,
# num_classes)` logits tensor: its result is a fresh `(batch, num_samples)` tensor of `dtype`
# (`int64` by default), never the logits themselves. A sampling loop feeds the draw back as the next
# step's input, so a result aliasing the logits would carry their shape and grow a dimension per
# round through the lookup.
import tensorflow as tf


def consume_default(s):
    assert s.shape == (2, 3) and s.dtype == tf.int64
    return s


def consume_int32(s):
    assert s.shape == (2, 1) and s.dtype == tf.int32
    return s


def consume_fed_back(prev):
    assert prev.shape in [(2, 3), (2, 1)] and prev.dtype == tf.int32
    return prev


logits = tf.random.uniform((2, 10))
consume_default(tf.random.categorical(logits, 3))
consume_int32(tf.random.categorical(logits, num_samples=1, dtype=tf.int32))

embedding = tf.random.uniform((10, 10))
prev = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
for i in range(3):
    consume_fed_back(prev)
    hidden = tf.gather(embedding, prev)
    step_logits = hidden[:, -1, :]
    prev = tf.random.categorical(step_logits, num_samples=1, dtype=tf.int32)
