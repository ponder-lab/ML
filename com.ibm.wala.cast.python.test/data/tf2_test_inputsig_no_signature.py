import tensorflow as tf


@tf.function
def f(x):
    # Decorated, no input_signature: traced on the concrete argument, so the
    # parameter takes the argument's shape and dtype.
    assert x.dtype == tf.int32
    assert x.shape.as_list() == [3]
    return x


f(tf.constant([1, 2, 3], dtype=tf.int32))
