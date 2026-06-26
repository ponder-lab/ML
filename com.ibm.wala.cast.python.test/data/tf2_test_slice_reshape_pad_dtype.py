import tensorflow as tf


def consume(x):
    pass


# Distilled from MusicTransformer's `RelativeGlobalAttention._skewing` (wala/ML#602).
# The slice receiver is `reshape(pad(tensor))`, a two-op dtype-preserving chain. The
# slice's shape recovers but its dtype lands at ⊤ unless `dtypesFromSSAChain` recurses
# through the `pad` and `reshape` ops back to the concrete-float32 input. Neither op
# alone reproduces; only the `reshape(pad(x))` chain does.
def skewing(tensor):
    padded = tf.pad(tensor, [[0, 0], [0, 0], [0, 0], [1, 0]])
    reshaped = tf.reshape(padded, [1, 1, 3, 2])
    return reshaped[:, :, 1:, :]


consume(skewing(tf.constant([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=tf.float32)))
