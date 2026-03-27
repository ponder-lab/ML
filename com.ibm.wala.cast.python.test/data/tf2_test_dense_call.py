import tensorflow as tf


def consume(tensor):
    pass


inputs = tf.keras.Input(shape=(3,))
layer = tf.keras.layers.Dense(4)
x = layer(inputs)

# Assertions for shape and dtype
assert x.shape.as_list() == [
    None,
    4,
], f"Expected shape [None, 4], got {x.shape.as_list()}"
assert x.dtype == tf.float32, f"Expected dtype float32, got {x.dtype}"

consume(x)
