# Regression test for `wala/ML#447`: `tf.constant(<bool>)` and
# `tf.constant([<bool>, ...])` exercise the `Boolean` arm of
# `TensorGenerator.getDTypesOfValue`. Without that arm, dtype inference
# threw `IllegalStateException: Unknown constant type: class java.lang.Boolean`.
import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


x = tf.constant(True)
assert x.shape == ()
assert x.dtype == tf.bool
f(x)

y = tf.constant([True, False, True])
assert y.shape == (3,)
assert y.dtype == tf.bool
g(y)
