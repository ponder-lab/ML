# Test for wala/ML#1111: reading `trainable_variables` off an instance of a subclassed Keras
# model gives a list, so `tape.gradient` over it returns a list of gradients (wala/ML#464).
import tensorflow as tf


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(2)

    def call(self, x):
        return self.dense(x)


class Plain(tf.keras.Model):

    def call(self, x):
        return x * 2.0


def take_sources(sources):
    assert isinstance(sources, list)
    return sources


def take_grads(grads):
    assert isinstance(grads, list)
    return grads


def take_plain_sources(sources):
    assert isinstance(sources, list)
    return sources


model = Net()
optimizer = tf.keras.optimizers.SGD(0.1)


def train_step(x):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(model(x))
    sources = take_sources(model.trainable_variables)
    grads = take_grads(tape.gradient(loss, sources))
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


train_step(tf.ones((3, 4)))

plain = Plain()
plain(tf.ones((3, 4)))
take_plain_sources(plain.trainable_variables)


def take_functional_sources(sources):
    assert isinstance(sources, list) and len(sources) == 2
    return sources


inp = tf.keras.Input(shape=(4,))
functional = tf.keras.Model(inp, tf.keras.layers.Dense(2)(inp))
take_functional_sources(functional.trainable_variables)
