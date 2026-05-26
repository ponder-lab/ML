import tensorflow as tf


def consume(x):
    pass


# wala/ML#509 regression guard: a user-defined class that happens to define
# `set_shape` must not be classified as a tensor by the static analysis.
# The `set_shape` recognition path—regardless of whether it's implemented
# Java-side (IR pattern matching) or XML-side (class-method dispatch)—must
# restrict pinning to actual tensor types and let non-tensor receivers
# fall through untouched.
class FakeShape:
    def set_shape(self, shape):
        return self


fake = FakeShape()
fake.set_shape([1, 2, 3])
consume(fake)

# Tensor-typed call kept in the same module so that `tf` is non-vacuous;
# without this the script analyzer doesn't load any TF modeling, which
# changes the call-graph shape relative to the other tensor fixtures.
t = tf.constant([1.0, 2.0, 3.0])
assert t.shape == (3,)
assert t.dtype == tf.float32
