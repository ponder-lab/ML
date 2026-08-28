# Test that the transposed-convolution and padding layers DECLINE rather than guess when their
# window arguments are not statically single-valued (https://github.com/wala/ML/issues/840).
# The tuple spellings below are the ones ordinary generator code actually uses.
import tensorflow as tf


def consume_tuple_window(a):
    pass


def consume_tuple_padding(b):
    pass


def consume_output_padding(c):
    pass


seed = tf.ones([2, 7, 7, 256])

# A tuple `kernel_size` and `strides` name the two axes separately and do not resolve, so the
# spatial extents degrade. The channel axis still resolves from `filters`, which is the point of
# resolving the two independently.
tuple_window = tf.keras.layers.Conv2DTranspose(
    128, (5, 5), strides=(2, 2), padding="same"
)(seed)
assert tuple_window.shape == (2, 14, 14, 128), tuple_window.shape
consume_tuple_window(tuple_window)

# A tuple `padding` names the two axes separately and does not resolve.
tuple_padding = tf.keras.layers.ZeroPadding2D(padding=(2, 3))(seed)
assert tuple_padding.shape == (2, 11, 13, 256), tuple_padding.shape
consume_tuple_padding(tuple_padding)

# A supplied `output_padding` shifts the extents by an amount the analysis does not read, so the
# spatial axes degrade rather than being reported from arithmetic that no longer describes them.
output_padding = tf.keras.layers.Conv2DTranspose(
    64, 3, strides=2, padding="same", output_padding=1
)(seed)
assert output_padding.shape == (2, 14, 14, 64), output_padding.shape
consume_output_padding(output_padding)
