import tensorflow as tf
from random import random


def func(t):
    pass


n = random()

a = None

if n > 0.5:
    a = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert a.shape == (3, 2)
else:
    a = tf.constant([[1.0], [3.0]])
    assert a.shape == (2, 1)

assert a.shape == (3, 2) or a.shape == (2, 1)
func(a)
