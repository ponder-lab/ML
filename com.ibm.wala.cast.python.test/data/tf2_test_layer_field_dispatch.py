# Dispatch probe (wala/ML#868 follow-up): two signature-identical layer classes,
# one instance constructed in the holder's own frame, one arriving through a
# constructor parameter, both stored to the same field and called through it
# with a keyword. At runtime both dispatch identically.
import tensorflow as tf


class Direct(tf.keras.layers.Layer):
    def call(self, inputs, k=None):
        return tf.reshape(inputs, (-1,))


class Passed(tf.keras.layers.Layer):
    def call(self, inputs, k=None):
        return tf.reshape(inputs, (-1,))


class Holder:
    def __init__(self, layer):
        if layer is None:
            self.layer = Direct()
        else:
            self.layer = layer
        self.k = 2

    def use(self, x):
        return self.layer(x, k=self.k)


h1 = Holder(None)
h2 = Holder(Passed())
x = tf.ones((3, 2))
assert h1.use(x).shape == (6,)
assert h2.use(x).shape == (6,)
