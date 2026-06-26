import tensorflow as tf


def consume(z):
    pass


# `tf.ones(x.shape)` takes its shape from another tensor's `.shape`. The shape
# argument is a `.shape` property read whose points-to set is empty, so resolution
# falls to `getDefaultShapes`. Regression guard for wala/ML#604: recover the shape
# from the source tensor (`x` has shape `(2, 2)`) rather than dropping to ⊤.
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
consume(tf.ones(x.shape))
