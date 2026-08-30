import tensorflow as tf

# wala/ML#832: a convolution stride SUPPLIED through a value the points-to substrate cannot
# represent (a concatenated list's element mints nothing, the wala/ML#805 class). Reading the
# empty resolution as an unsupplied argument would patch in the API default of one and compose
# the default's shape for a program that runs stride two; the honest answer degrades the spatial
# extents while the rank and the filter count survive.


def consume_hidden(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (1, 4, 4, 6), x.shape
    assert x.dtype == tf.float32


hidden = ([2] + [1])[0]
model = tf.keras.Sequential(
    [tf.keras.layers.Conv2D(6, 3, strides=hidden, padding="same")]
)
consume_hidden(model(tf.ones((1, 8, 8, 3))))


def consume_starred(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (1, 8, 8, 6), x.shape
    assert x.dtype == tf.float32


# A starred constructor: positional alignment past the unpack is unreliable, so the detection
# cannot tell whether a stride was supplied, and indeterminate declines. The cost is stated: this
# program genuinely omits the stride, and the honest price of not re-opening the default patch
# for invisible supplies is losing its window here.
cfg = (6, 3)
starred = tf.keras.Sequential([tf.keras.layers.Conv2D(*cfg, padding="same")])
consume_starred(starred(tf.ones((1, 8, 8, 3))))
