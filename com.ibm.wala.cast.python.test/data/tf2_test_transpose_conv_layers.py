# Test that `Conv2DTranspose` and `ZeroPadding2D` compute their output shapes instead of
# dropping them for the rest of the chain (https://github.com/wala/ML/issues/840).
import tensorflow as tf


def consume_upsampled(a):
    pass


def consume_padded(b):
    pass


def consume_chained(c):
    pass


seed = tf.ones([2, 7, 7, 256])

# `padding='same'`: each spatial extent is scaled by the stride, and the channel axis becomes the
# declared filter count.
upsampled = tf.keras.layers.Conv2DTranspose(128, 5, strides=2, padding="same")(seed)
assert upsampled.shape == (2, 14, 14, 128), upsampled.shape
assert upsampled.dtype == tf.float32, upsampled.dtype
consume_upsampled(upsampled)

# The default padding of one grows each spatial extent by two, one row or column per side.
padded = tf.keras.layers.ZeroPadding2D()(seed)
assert padded.shape == (2, 9, 9, 256), padded.shape
assert padded.dtype == tf.float32, padded.dtype
consume_padded(padded)

# The chain matters more than either layer alone: before this modeling, the first unmodeled layer
# dropped the shape for everything downstream of it.
chained = tf.keras.layers.ZeroPadding2D()(upsampled)
chained = tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding="same")(chained)
assert chained.shape == (2, 32, 32, 64), chained.shape
assert chained.dtype == tf.float32, chained.dtype
consume_chained(chained)
