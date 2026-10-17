# Test for wala/ML#1119: applying a `BatchNormalization` layer registers the update of its moving
# statistics with `add_update`, a write to the layer that a mod-ref query sees in the `call` that
# applies it, and the layer receives the `training` value it is applied with.
import tensorflow as tf


class Block(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        return self.bn(x, training=training)


class Counter(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.v = tf.Variable(0.0, trainable=False)

    def call(self, x):
        self.add_update(lambda: self.v.assign_add(1.0))
        return x


class Recurrent(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.rnn = tf.keras.layers.SimpleRNN(4, stateful=True)

    def call(self, x):
        return self.rnn(x)


class Plain(tf.keras.layers.Layer):
    def call(self, x):
        return x * 2.0


bn_train = tf.keras.layers.BatchNormalization()
bn_infer = tf.keras.layers.BatchNormalization()


def normalize_training(t):
    return bn_train(t, training=True)


def normalize_inference(t):
    return bn_infer(t, training=False)


x = tf.random.uniform([8, 4]) + 3.0
block = Block()
block(x, training=True)
assert abs(float(block.bn.moving_mean.numpy()[0])) > 0.0
counter = Counter()
counter(x)
assert float(counter.v.numpy()) == 1.0
Recurrent()(tf.ones([2, 3, 4]))
Plain()(x)
normalize_training(x)
assert abs(float(bn_train.moving_mean.numpy()[0])) > 0.0
normalize_inference(x)
assert float(bn_infer.moving_mean.numpy()[0]) == 0.0
