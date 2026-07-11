import tensorflow as tf


def consume(t):
    pass


def get_shape_list(tensor):
    return tensor.shape.as_list()


config = {"factor": 2}


# Single-member counterpart of tf2_test_embedding_dynamic_size.py
# (wala/ML#717): dimension arithmetic over a plain (non-φ) shape-vector
# subscript with a config-sourced factor degrades that element's value to
# dynamic while the rank and the literal element survive.
def widen(t):
    factor = config.get("factor", 2)
    shape = get_shape_list(t)
    return tf.reshape(t, [shape[0] * factor, 6])


out = widen(tf.ones((4, 12)))

assert out.shape == (8, 6)
assert out.dtype == tf.float32

consume(out)
