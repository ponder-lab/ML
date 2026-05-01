import tensorflow as tf
from random import random


def func(t):
    pass


n = random()

if n > 0.5:
    l = [[1.0], [3.0]]
else:
    l = [1.0, 3.0]

a = tf.constant(l)
assert a.shape == (2, 1) or a.shape == (2,)

func(a)
