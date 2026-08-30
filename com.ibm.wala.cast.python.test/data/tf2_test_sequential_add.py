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


def consume_returned_then_added(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (3, 2)
    assert x.dtype == tf.float32


def consume_popped(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (3, 4)
    assert x.dtype == tf.float32


# A builder that returns its stage, with the CALLER adding afterward. The frame that constructs
# the model sees one layer; the model the program runs has two. Following the return is what keeps
# the builder idiom above composable while stopping this.
def half_built():
    m = tf.keras.Sequential()
    m.add(tf.keras.layers.Dense(4))
    return m


returned_then_added = half_built()
returned_then_added.add(tf.keras.layers.Dense(2))
consume_returned_then_added(returned_then_added(tf.ones((3, 5))))

# `pop()` shrinks the chain. A walk that tolerated any member read whose name is a constant would
# compose both layers and report the one the program removed.
popped = tf.keras.Sequential()
popped.add(tf.keras.layers.Dense(4))
popped.add(tf.keras.layers.Dense(2))
popped.pop()
consume_popped(popped(tf.ones((3, 5))))


def consume_after_model_methods(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (3, 2)
    assert x.dtype == tf.float32


# The whole-model methods a training script calls on the model it built. None of them changes the
# layer list, which is why the walk tolerates them, and this arm is what would catch it if one did:
# a member that quietly grew or shrank the chain would make the composed shape disagree with the
# runtime assert above.
with_methods = tf.keras.Sequential()
with_methods.add(tf.keras.layers.Dense(4))
with_methods.add(tf.keras.layers.Dense(2))
with_methods.build(input_shape=(None, 5))
with_methods.compile(optimizer="sgd", loss="mse")
with_methods.summary()
with_methods.count_params()
with_methods.get_weights()
consume_after_model_methods(with_methods(tf.ones((3, 5))))
