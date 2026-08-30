# wala/ML#854's remaining half: a `Sequential` grown one layer at a time by `.add()` keeps the same
# information as a constructor list, but nowhere the analysis can read it as a container. The
# layers arrive one call at a time, and their ORDER, which the whole composition depends on, exists
# only as the order of the calls in the building code.
import tensorflow as tf


def consume_plain(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (3, 2)
    assert x.dtype == tf.float32


def consume_conditional_on(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (1, 4, 4, 6)
    assert x.dtype == tf.float32


def consume_conditional_off(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (1, 4, 4, 6)
    assert x.dtype == tf.float32


def consume_looped(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (3, 2)
    assert x.dtype == tf.float32


# A straight-line builder. Two layers of the same class again, so a fold that resolved them
# through the model call's node rather than per layer would report (3, 4).
plain = tf.keras.Sequential()
plain.add(tf.keras.layers.Dense(4))
plain.add(tf.keras.layers.Dense(2))
consume_plain(plain(tf.ones((3, 5))))


# The canonical downsampling stage: a normalization added only when it is asked for. The two paths
# are two different models, so the composition is a union over both rather than a choice between
# them. Here both paths agree on the shape, since the conditional layer preserves it; that is a
# fact about this stage rather than about the mechanism.
def downsample(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding="same", use_bias=False)
    )
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


consume_conditional_on(downsample(6, 4)(tf.ones((1, 8, 8, 3))))
consume_conditional_off(downsample(6, 4, apply_batchnorm=False)(tf.ones((1, 8, 8, 3))))


# A loop that adds. The chain's LENGTH is a trip count rather than a static list, so the walk has
# to decline: composing the body once would report a one-layer model where the program builds two.
looped = tf.keras.Sequential()
for width in [4, 2]:
    looped.add(tf.keras.layers.Dense(width))
consume_looped(looped(tf.ones((3, 5))))


def consume_union(x):
    assert isinstance(x, tf.Tensor)
    # The two paths disagree, so the sink sees one shape per call and the static answer is the
    # union of both. A fold that composed only one path would report a single member here.
    assert x.shape in [(3, 4), (3, 7)]
    assert x.dtype == tf.float32


def maybe_wider(wide):
    m = tf.keras.Sequential()
    m.add(tf.keras.layers.Dense(4))
    if wide:
        m.add(tf.keras.layers.Dense(7))
    return m


consume_union(maybe_wider(True)(tf.ones((3, 5))))
consume_union(maybe_wider(False)(tf.ones((3, 5))))
