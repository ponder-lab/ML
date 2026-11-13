import tensorflow as tf

SEEN = []


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.d = tf.keras.layers.Dense(1)

    def call(self, x, training=None):
        SEEN.append(training)
        return self.d(x)


model = Net()
model.predict(tf.ones([8, 4]), verbose=0)
assert set(SEEN) == {False}, SEEN
