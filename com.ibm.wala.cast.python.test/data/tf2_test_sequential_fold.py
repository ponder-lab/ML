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


def consume_sliced(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (3, 2)
    assert x.dtype == tf.float32


def consume_appended(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (3, 2)
    assert x.dtype == tf.float32


def consume_nested(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (3, 6)
    assert x.dtype == tf.float32


# A list the caller DERIVED rather than wrote. The slice application's result aliases its
# receiver, so the constructor's argument reaches the analysis as the full list's allocation with
# a catalog that is contiguous and single-valued at every position; it is simply the wrong list.
# Folding it would compose the layer the slice removed and report (3, 7).
base = [
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Dense(7),
]
sliced = tf.keras.Sequential(base[:2])
consume_sliced(sliced(tf.ones((3, 5))))

# A literal the program appended to. The appended layer lands under the synthetic append-contents
# field rather than at an integer index, so the catalog shows one layer and reads as a complete
# run. Folding it would stop at the literal's layer and report (3, 4).
appended = [tf.keras.layers.Dense(4)]
appended.append(tf.keras.layers.Dense(2))
appended_model = tf.keras.Sequential(appended)
consume_appended(appended_model(tf.ones((3, 5))))

# A `Sequential` nested inside another at a non-leading position. This one composes: the inner
# model's instance type dispatches back through the same table, and the inner fold reads the
# running shape through the same seam. The correctness rides on the seam's index matching the
# model call's `inputs` position, which nothing else pins.
nested = tf.keras.Sequential(
    [tf.keras.layers.Dense(4), tf.keras.Sequential([tf.keras.layers.Dense(6)])]
)
consume_nested(nested(tf.ones((3, 5))))
