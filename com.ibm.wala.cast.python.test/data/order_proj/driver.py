import tensorflow as tf

from alpha_sub import Outer, Sub, scale_inside

a = Sub().scale(tf.ones((2, 3)))
assert a.shape == (2, 3)

b = scale_inside(tf.ones((4,)))
assert b.shape == (4,)

c = Outer.Nested().scale(tf.ones((5, 6, 7)))
assert c.shape == (5, 6, 7)
