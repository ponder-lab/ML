import tensorflow as tf


def g(b):
    pass


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.int32)])
def f(x):
    # `input_signature` governs `x`, so at `g`'s call site the argument is (None,) int32,
    # NOT the (3,) of the value passed to `f` below.
    assert x.shape.as_list() == [None]
    assert x.dtype == tf.int32
    g(x)


f(tf.constant([1, 2, 3], dtype=tf.int32))
