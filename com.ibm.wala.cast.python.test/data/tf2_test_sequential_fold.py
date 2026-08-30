# wala/ML#832: a `Sequential` keeps its forward chain in its LAYER LIST, where a functional model
# keeps it in `inputs`/`outputs`, so the machinery that walks a functional model backwards had
# nothing to anchor on and the call's shape floored at unknown. These arms exercise the fold that
# composes the list, and the two conditions under which it must refuse rather than guess.
import tensorflow as tf


def consume_two_dense(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (3, 2)
    assert x.dtype == tf.float32


def consume_conv_stack(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (1, 8, 8, 6)
    assert x.dtype == tf.float32


def consume_unmodeled(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (3, 5)
    assert x.dtype == tf.float32


# Two layers of the SAME class in one model. Each needs its own `units`, so a fold that resolved
# the layers through anything keyed on the model call's node would read one layer's answer for
# both and report (3, 4).
two_dense = tf.keras.Sequential([tf.keras.layers.Dense(4), tf.keras.layers.Dense(2)])
consume_two_dense(two_dense(tf.ones((3, 5))))

# A convolutional stage at the shape a residual shortcut uses: the window folds the spatial
# extents and sets the channel, and the two layers after it preserve what it produced.
conv_stack = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            6, kernel_size=3, strides=1, padding="same", use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ]
)
consume_conv_stack(conv_stack(tf.ones((1, 8, 8, 3))))


# A layer the analysis does not model. The fold must abandon the whole composition here: applying
# the layers it DOES know would report the chain's shape as though the unknown layer were the
# identity, which is a confidently wrong shape rather than a missing one.
class Unmodeled(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs


unmodeled = tf.keras.Sequential([Unmodeled()])
consume_unmodeled(unmodeled(tf.ones((3, 5))))
