# Test https://github.com/wala/ML/issues/1249: `fit` and `predict` call the model with `training`
# `True` and `False`, and a layer the model's `call` calls without `training` takes each.
import tensorflow as tf

SEEN = []


class Inner(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.d = tf.keras.layers.Dense(1)

    def call(self, x, training=None):
        SEEN.append(training)
        return self.d(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.inner = Inner()

    def call(self, x, training=None):
        return self.inner(x)


model = Net()
model.compile(optimizer="sgd", loss="mse")
model.fit(tf.ones([8, 4]), tf.ones([8, 1]), epochs=1, verbose=0)
model.predict(tf.ones([8, 4]), verbose=0)
assert set(SEEN) == {True, False}, SEEN
