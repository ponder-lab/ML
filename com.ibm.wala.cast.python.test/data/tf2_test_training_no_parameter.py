import tensorflow as tf


def model_input(x):
    assert x.shape == (None, 4)
    assert x.dtype == tf.float32
    return x


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.d = tf.keras.layers.Dense(1)

    def call(self, x):
        return self.d(model_input(x))


model = Net()
model.compile(optimizer="sgd", loss="mse")
model.fit(tf.ones([8, 4]), tf.ones([8, 1]), epochs=1, verbose=0)
