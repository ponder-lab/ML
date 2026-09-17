import tensorflow as tf

from a import A
from b import make

r = A().scale(tf.ones((2, 3)))
assert r.shape == (2, 3)

s = make()().scale(tf.ones((4, 5)))
assert s.shape == (4, 5)
