# A Keras Model whose `call` returns a tuple, unpacked at the call site, against one returning a
# single tensor. Both returns are the same shape, so a difference between the two sinks is about the
# unpacking rather than about what `call` computes.
import tensorflow as tf


def consume_single(a):
    pass


def consume_first(a):
    pass


def consume_second(a):
    pass


class SingleModel(tf.keras.Model):
    def call(self, x):
        return tf.reshape(x, [2, 20])


class TupleModel(tf.keras.Model):
    def call(self, x):
        return tf.reshape(x, [2, 20]), tf.reshape(x, [2, 20])


inp = tf.ones((40,))

single = SingleModel()(inp)
assert single.shape == (2, 20)
consume_single(single)

first, second = TupleModel()(inp)
assert first.shape == (2, 20)
assert second.shape == (2, 20)
consume_first(first)
consume_second(second)
