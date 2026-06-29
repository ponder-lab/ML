import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.int32)])
def f(x):
    # Decorated with a signature, traced (default): the signature governs the
    # parameter, so its shape is dynamic (None,) -- NOT the argument's (3,).
    assert x.dtype == tf.int32
    assert x.shape.as_list() == [None]
    return x


f(tf.constant([1, 2, 3], dtype=tf.int32))
