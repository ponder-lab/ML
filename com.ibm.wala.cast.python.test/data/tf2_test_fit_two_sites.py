# Test https://github.com/wala/ML/issues/997: one model class fit at two sites on datasets of
# different element types, so the step's inputs are either dataset's.
import tensorflow as tf


def consume_x(x):
    assert (x.shape[1:], x.dtype) in (((4,), tf.float32), ((3,), tf.int32))
    return x


class Net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(tf.cast(inputs, tf.float32))

    def train_step(self, data):
        x, y = data
        consume_x(x)
        return {}


floats = tf.data.Dataset.from_tensor_slices((tf.ones([6, 4]), tf.zeros([6]))).batch(2)
ints = tf.data.Dataset.from_tensor_slices(
    (tf.zeros([6, 3], tf.int32), tf.zeros([6]))
).batch(2)
model = Net()
model.compile(optimizer="sgd", loss="mse")
model.fit(floats, epochs=1, verbose=0)
model.fit(ints, epochs=1, verbose=0)
