# Test https://github.com/wala/ML/issues/996: a keyword argument at a call to a summarized method
# binds the formal of that name, so the pass-through layer's output carries the input's type.
import tensorflow as tf


def consume(y):
    assert y.shape == (4, 4) and y.dtype == tf.float32
    return y


layer = tf.keras.layers.Dropout(0.5)
consume(layer(inputs=tf.ones((4, 4))))
