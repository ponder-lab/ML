import tensorflow as tf


def g(b):
    pass


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.int32)])
def f(x):
    # Traced (the default): `input_signature` governs `x`, so `g` receives (None,) int32.
    assert x.shape.as_list() == [None]
    assert x.dtype == tf.int32
    g(x)


# Under `run_functions_eagerly` the signature would be ignored and `g` would instead receive this
# argument's (3,) int32. A static analysis cannot know the execution mode, so the sound type of
# `g`'s parameter is the set {(None,), (3,)} int32.
arg = tf.constant([1, 2, 3], dtype=tf.int32)
assert arg.shape == (3,)
assert arg.dtype == tf.int32
f(arg)
