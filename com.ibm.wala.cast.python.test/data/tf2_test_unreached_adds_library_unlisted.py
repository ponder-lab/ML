# A function the call graph does not reach may still run. Its `tf.backend.update` is no TensorFlow
# function the analysis knows to resize nothing, so it may add an element to any dict.
import tensorflow as tf


def consume(c):
    pass


def apply_moment(m, grad):
    return tf.backend.update(m, 0.9 * m + grad)


c = {"a": 1, "b": 2}
assert len(c) == 2 and c["a"] == 1
consume(c)
