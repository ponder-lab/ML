# A function the call graph does not reach may still run. Its `tf.backend.update` is a TensorFlow
# function no summary models, and TensorFlow is registered as a library whose functions resize no
# container passed to them, so it adds to no dict (wala/ML#1182).
import tensorflow as tf


def consume(c):
    pass


def apply_moment(m, grad):
    return tf.backend.update(m, 0.9 * m + grad)


c = {"a": 1, "b": 2}
assert len(c) == 2 and c["a"] == 1
consume(c)
