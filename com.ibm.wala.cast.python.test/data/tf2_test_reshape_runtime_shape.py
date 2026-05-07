# Regression guard for `tf.reshape(x, tf.shape(y))` shape inference (wala/ML#489).
#
# At runtime, `z` has shape (2, 3) of float32. The analyzer currently produces a malformed
# compound-dim shape (`[Compound,[Constant,0, Constant,0, Constant,0], ...]`) instead of
# either the precise (2, 3) answer or a sound ⊤ (null shape). The corresponding JUnit test
# is suppressed with `@Test(expected = AssertionError.class)` and a TODO referencing #489;
# when modeling lands, flip the suppression to plain `@Test`.
import tensorflow as tf


def f(a):
    pass


x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # shape (2, 3)
z = tf.reshape(x, tf.shape(y))  # runtime: shape (2, 3) of float32
assert isinstance(z, tf.Tensor)
assert z.shape == (2, 3)
assert z.dtype == tf.float32
f(z)
