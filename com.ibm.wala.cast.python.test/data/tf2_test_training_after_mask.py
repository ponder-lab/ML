import tensorflow as tf

SEEN = []


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.d = tf.keras.layers.Dense(1)

    def call(self, x, mask=None, training=None):
        SEEN.append((mask, training))
        return self.d(x)


model = Net()
model.compile(optimizer="sgd", loss="mse")
model.fit(tf.ones([8, 4]), tf.ones([8, 1]), epochs=1, verbose=0)
assert set(SEEN) == {(None, True)}, SEEN
