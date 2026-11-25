# A function the call graph does not reach may still run, so a call there that may add an element
# to a container it cannot name may add one to any container. This one cannot: it is a TensorFlow
# function that assigns a variable. The container consumed is a dict, which an `update` may reach,
# since no list has the method (https://github.com/wala/ML/issues/1328).
import tensorflow as tf


def consume(c):
    assert len(c) == 2 and c["a"] == 1


def apply_moment(m, grad):
    return tf.keras.backend.update(m, 0.9 * m + grad)


consume({"a": 1, "b": 2})
