# Test https://github.com/wala/ML/issues/740. Two call sites of the same
# method with equal total arity but different positional/keyword splits.
import tensorflow as tf


class C:

    def f(self, x, training=True):
        assert isinstance(x, tf.Tensor)


c = C()

a = tf.ones([1, 2])
assert a.shape == (1, 2)
assert a.dtype == tf.float32

b = tf.ones([1, 3])
assert b.shape == (1, 3)
assert b.dtype == tf.float32

c.f(a, False)
c.f(training=False, x=b)
