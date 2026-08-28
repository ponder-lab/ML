# Test that `Conv1D` is modeled and that `Conv2D` folds its spatial extents instead of reporting
# them unresolved (https://github.com/wala/ML/issues/840).
import tensorflow as tf


def consume_conv1d(a):
    pass


def consume_conv2d_valid(b):
    pass


def consume_conv2d_same(c):
    pass


def consume_padded_then_conv(d):
    pass


steps = tf.ones([4, 20, 8])

# `valid` padding, the default: the temporal axis loses the kernel's span.
conv1d = tf.keras.layers.Conv1D(16, 5)(steps)
assert conv1d.shape == (4, 16, 16), conv1d.shape
assert conv1d.dtype == tf.float32, conv1d.dtype
consume_conv1d(conv1d)

image = tf.ones([4, 32, 32, 3])

# `valid` padding at stride 2.
conv2d_valid = tf.keras.layers.Conv2D(64, 4, strides=2)(image)
assert conv2d_valid.shape == (4, 15, 15, 64), conv2d_valid.shape
consume_conv2d_valid(conv2d_valid)

# `same` padding scales the extent by the stride rather than shrinking it by the kernel.
conv2d_same = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(image)
assert conv2d_same.shape == (4, 16, 16, 32), conv2d_same.shape
consume_conv2d_same(conv2d_same)

# The chain is the point: a padding layer's resolved extent used to die at the next convolution,
# because the convolution reported both spatial axes unresolved whatever it was given.
padded = tf.keras.layers.ZeroPadding2D()(image)
padded_then_conv = tf.keras.layers.Conv2D(8, 4, strides=1)(padded)
assert padded_then_conv.shape == (4, 31, 31, 8), padded_then_conv.shape
consume_padded_then_conv(padded_then_conv)


def consume_tuple_kernel(e):
    pass


def consume_dilated(f):
    pass


# A tuple `kernel_size` names each spatial axis separately and does not resolve, so the extents
# degrade while the channel axis still resolves. This is the spelling most generator code writes.
tuple_kernel = tf.keras.layers.Conv2D(16, (3, 3), strides=2)(image)
assert tuple_kernel.shape == (4, 15, 15, 16), tuple_kernel.shape
consume_tuple_kernel(tuple_kernel)

# Dilation widens the kernel's span, so it is read rather than assumed: a kernel of 3 dilated by 2
# spans 5, and `valid` padding loses that span rather than the kernel size.
dilated = tf.keras.layers.Conv1D(12, 3, dilation_rate=2)(steps)
assert dilated.shape == (4, 16, 12), dilated.shape
consume_dilated(dilated)
