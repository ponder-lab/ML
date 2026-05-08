# Regression guard for `tf.reshape(x, tf.shape(y))` shape inference (wala/ML#489).
#
# At runtime, `z` has shape (2, 3) of float32. Post the wala/ML#489 root-cause fix on this
# PR's `tensorflow.xml` (separating `tf.shape`'s allocation from `pass_through` aliasing),
# `tf.shape(y)` allocates a fresh `Ltensorflow/python/framework/ops/Tensor` whose static
# shape `getShapesFromShapeArgument` doesn't recognize. `Reshape` doesn't localize the
# resulting `IllegalStateException` (unlike `BroadcastTo`), per this PR's keep-throw-
# everywhere-except-BroadcastTo design, so the exception aborts the analysis on this fixture.
#
# The corresponding JUnit test (`testReshapeRuntimeShape`) is annotated
# `@Test(expected = IllegalStateException.class)` with a TODO referencing #489. When `Reshape`
# gains a precise composer or a localized try/catch, the suppression lifts and the test
# starts asserting the precise shape (or ⊤) instead.
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
