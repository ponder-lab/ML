import tensorflow as tf


def consume(t):
    pass


def consume2(t):
    pass


xs = [tf.ones((4, 8))]
ys = [tf.zeros((2, 2))]

for p in zip(xs, ys):
    consume(p[0])

for x, y in zip(xs, ys):
    consume2(x)
