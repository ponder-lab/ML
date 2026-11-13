import tensorflow as tf

SEEN = []


class Block(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.d = tf.keras.layers.Dense(1)

    def call(self, x, training=None):
        SEEN.append(training)
        return self.d(x)


model = tf.keras.Sequential([Block()])
model.compile(optimizer="sgd", loss="mse")
model.fit(tf.ones([8, 4]), tf.ones([8, 1]), epochs=1, verbose=0)
model.predict(tf.ones([8, 4]), verbose=0)
assert set(SEEN) == {True, False}, SEEN
