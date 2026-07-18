import numpy as np
import tensorflow as tf


def consume(t):
    pass


# Content-dependent: the shape flows from an opaque file loader. A genuine wala/ML#370 candidate.
# Analyzed statically (the loaded file need not exist), like the vendored dataset fixtures.
loaded = np.load("data.npy")
a = tf.zeros(loaded.shape)
consume(a)

# Recoverable gap: the shape flows from a modeled op (transpose) whose shape the analyzer missed
# for a non-content reason (a runtime-computed perm), not from an opaque source.
x = tf.ones((2, 3, 4))
perm = tf.random.shuffle(tf.range(3))
t = tf.transpose(x, perm)
b = tf.zeros(t.shape)
consume(b)
