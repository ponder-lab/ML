import tensorflow as tf

tf.config.run_functions_eagerly(True)


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.int32)])
def f(x):
    # Same decorated-with-signature function as the traced case, but under
    # run_functions_eagerly the signature is ignored: the parameter takes the
    # argument's concrete shape (3,), not the signature's (None,).
    assert x.dtype == tf.int32
    assert x.shape.as_list() == [3]
    return x


f(tf.constant([1, 2, 3], dtype=tf.int32))
