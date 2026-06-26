import tensorflow as tf


def consume(v):
    pass


# `tf.math.top_k(x, k)` returns values/indices of shape `x.shape[:-1] + (k,)`.
# Regression guard for wala/ML#609: `values` of `top_k((4,) input, k=2)` is `(2,)`
# float32, composed from the input shape and k rather than left at ⊤.
values, indices = tf.math.top_k(tf.constant([1.0, 3.0, 2.0, 5.0]), k=2)
consume(values)
