import tensorflow as tf
import random


def add(a, b):
    return a + b


if random.random() < 0.5:
    a = 1
else:
    a = 3

c = add(tf.ones([a, 2]), tf.ones([2, 2]))
