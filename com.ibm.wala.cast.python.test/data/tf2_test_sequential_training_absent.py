import tensorflow as tf

SEEN = []


def scale(x):
    return x * 2.0


class Block(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.d = tf.keras.layers.Dense(4)

    def call(self, x, training=False):
        SEEN.append(("Block", training))
        if training:
            x = scale(x)
        return self.d(x)


model = tf.keras.Sequential([Block()])
model(tf.ones([8, 4]))
assert set(SEEN) == {("Block", False)}, SEEN
