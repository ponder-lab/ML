# wala/ML#832: a list built only by appending in a loop tells the analysis WHICH layer the chain
# is made of and not HOW MANY of them. The rank of such a chain does not depend on the count as
# long as the layer preserves rank, so refusing outright would throw away the part the program's
# own text determines.
import tensorflow as tf


def consume_repeated(x):
    assert isinstance(x, tf.Tensor)
    # Rank 4 and the batch survive the repetition whatever the count is. The spatial extents do
    # not: each application halves them, so their value depends on how many times the loop ran.
    assert len(x.shape) == 4
    assert x.shape[0] == 1
    assert x.shape[3] == 3
    assert x.dtype == tf.float32


def consume_preserving(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (1, 8, 8, 3)
    assert x.dtype == tf.float32


def consume_rank_changing(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (1, 192)
    assert x.dtype == tf.float32


# A stage whose layers are appended in a loop. Every application halves the spatial extents, so
# those axes depend on the count and must degrade; the batch and the channel are fixed points of
# the transform and survive.
strided = []
for _ in range(2):
    strided.append(
        tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="valid")
    )
consume_repeated(tf.keras.Sequential(strided)(tf.ones((1, 8, 8, 3))))

# The same build with a layer that changes nothing: every axis is a fixed point, so every axis
# survives and the result is exact despite the count being unknown.
preserving = []
for _ in range(3):
    preserving.append(tf.keras.layers.BatchNormalization())
consume_preserving(tf.keras.Sequential(preserving)(tf.ones((1, 8, 8, 3))))

# A layer that changes the RANK. Applying it a second time is not even well defined for this
# input, so a chain of unknown length over it has no rank the analysis can name and must decline.
flattening = []
for _ in range(1):
    flattening.append(tf.keras.layers.Flatten())
consume_rank_changing(tf.keras.Sequential(flattening)(tf.ones((1, 8, 8, 3))))
