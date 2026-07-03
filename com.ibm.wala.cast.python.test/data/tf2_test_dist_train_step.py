# Probe for wala/ML#683: `self._distribution_strategy` is Keras-internal state (assigned by
# `Model.__init__`, never in user code); the dispatch of `run` on it must resolve for the
# strategy summary to materialize the invoke edge to the function argument.
import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        return inputs

    def __train_step(self, x, y):
        return x + y

    def dist_train_step(self, x, y):
        return self._distribution_strategy.run(self.__train_step, args=(x, y))


def consume(t):
    pass


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = MyModel()

x = tf.ones((2, 3))
y = tf.ones((2, 3))
out = model.dist_train_step(x, y)
consume(out)

assert out.shape == (2, 3)
assert out.dtype == tf.float32
