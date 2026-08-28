import tensorflow as tf


def consume_maxpool(x):
    pass


def consume_maxpool_defaults(x):
    pass


def consume_maxpool_alias(x):
    pass


def consume_avgpool(x):
    pass


def consume_globalmax1d(x):
    pass


def consume_globalavg1d_alias(x):
    pass


def consume_globalmax2d(x):
    pass


# Applied directly, without a surrounding model: the output shape follows from the input shape
# together with the pool size, strides, and padding stored on the layer (wala/ML#840).
pool = tf.keras.layers.MaxPool2D(2, strides=2)
pool_out = pool(tf.ones((32, 28, 28, 1)))
assert pool_out.shape == (32, 14, 14, 1), pool_out.shape
assert pool_out.dtype == tf.float32, pool_out.dtype
consume_maxpool(pool_out)

# Unsupplied strides default to the pool size.
pool_defaults = tf.keras.layers.MaxPool2D(2)
defaults_out = pool_defaults(tf.ones((32, 28, 28, 1)))
assert defaults_out.shape == (32, 14, 14, 1), defaults_out.shape
consume_maxpool_defaults(defaults_out)

# The alias spelling resolves to the same class.
pool_alias = tf.keras.layers.MaxPooling2D(2, strides=2)
alias_out = pool_alias(tf.ones((32, 28, 28, 1)))
assert alias_out.shape == (32, 14, 14, 1), alias_out.shape
consume_maxpool_alias(alias_out)

# Average pooling under `same` padding: the spatial extent is ceil(size / stride), where `valid`
# would give 13 for a pool of 3 over 28 at stride 2.
avg = tf.keras.layers.AveragePooling2D(3, strides=2, padding="same")
avg_out = avg(tf.ones((32, 28, 28, 1)))
assert avg_out.shape == (32, 14, 14, 1), avg_out.shape
consume_avgpool(avg_out)

# Global pooling drops the interior axes and keeps batch and features.
gmp1 = tf.keras.layers.GlobalMaxPooling1D()
gmp1_out = gmp1(tf.ones((32, 10, 64)))
assert gmp1_out.shape == (32, 64), gmp1_out.shape
consume_globalmax1d(gmp1_out)

gmp2 = tf.keras.layers.GlobalMaxPooling2D()
gmp2_out = gmp2(tf.ones((32, 28, 28, 3)))
assert gmp2_out.shape == (32, 3), gmp2_out.shape
consume_globalmax2d(gmp2_out)

# The `GlobalAvgPool1D` alias spelling resolves to the modeled `GlobalAveragePooling1D` class.
gap1 = tf.keras.layers.GlobalAvgPool1D()
gap1_out = gap1(tf.ones((32, 10, 64)))
assert gap1_out.shape == (32, 64), gap1_out.shape
consume_globalavg1d_alias(gap1_out)
