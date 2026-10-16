# Test for wala/ML#1098: a program class that subclasses a built-in Keras layer is a Keras layer, so
# its superclass chain reaches `Layer`, and a `call` it overrides is the one its instances run.
import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class D(tf.keras.layers.Dense):
    pass


class Scaled(tf.keras.layers.Dense):

    def call(self, inputs):
        assert inputs.shape == (2, 3) and inputs.dtype == tf.float32
        return super().call(inputs) * 2.0


y = D(4)(BatchNormalization()(tf.random.normal([2, 3])))
assert y.shape == (2, 4)

z = Scaled(4)(tf.random.normal([2, 3]))
assert z.shape == (2, 4)
