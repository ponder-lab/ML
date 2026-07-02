# Probe for the collection-dataflow family (wala/ML#570/#618/#659): a function returning a tuple
# of tensors, unpacked at the call site, keeps each element's tensor type.
import tensorflow as tf


def consume(t):
    pass


def produce():
    logits = tf.ones((4, 8))
    presents = tf.zeros((2, 2))
    return logits, presents


a, b = produce()
consume(a)
assert a.shape == (4, 8)
assert a.dtype == tf.float32
