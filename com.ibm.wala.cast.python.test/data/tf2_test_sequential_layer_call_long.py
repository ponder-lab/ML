import tensorflow as tf

# wala/ML#1089: a literal layer list longer than the positions the `Sequential` summary
# unrolls. The user layer sits at position 32, and the model is its only caller.


class Last(tf.keras.layers.Layer):

    def call(self, x):
        assert x.shape == (2, 3), x.shape
        assert x.dtype == tf.float32
        return x * 2.0


model = tf.keras.Sequential(
    [
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        tf.keras.layers.Activation("linear"),
        Last(),
    ]
)
assert model(tf.ones([2, 3])).shape == (2, 3)
