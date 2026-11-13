import tensorflow as tf

SEEN = []


class Block(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.d = tf.keras.layers.Dense(4)

    def call(self, x, training=False):
        SEEN.append(("Block", training))
        return self.d(x)


class Masked(tf.keras.layers.Layer):

    def call(self, inputs, mask=None, training=None):
        SEEN.append(("Masked", mask, training))
        return inputs


class Plain(tf.keras.layers.Layer):

    def call(self, x):
        SEEN.append(("Plain",))
        return x


model = tf.keras.Sequential([Block(), Masked(), Plain()])
model(tf.ones([8, 4]), training=True)
assert set(SEEN) == {("Block", True), ("Masked", None, True), ("Plain",)}, SEEN
