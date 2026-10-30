# A function the call graph does not reach may still run, so a call there that may add an element
# to a container it cannot name may add one to any container. This one cannot: it is a TensorFlow
# function that assigns a variable.
import tensorflow as tf


def consume(c):
    assert len(c) == 2 and c[0] == 1


def apply_moment(m, grad):
    return tf.keras.backend.update(m, 0.9 * m + grad)


consume([1, 2])
