# Test https://github.com/wala/ML/issues/977: a module inside the path, imported from a script
# outside every path entry.
import tensorflow as tf


def f(x):
    return tf.add(x, 1)
