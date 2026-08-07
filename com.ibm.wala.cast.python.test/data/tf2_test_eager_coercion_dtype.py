# wala/ML#828: a parameter's eager-effective dtype is the one its consumers coerce it to, not the
# one its callers feed it. TensorFlow converts a NumPy argument through the dtype of the operand
# beside it, so the two disagree exactly when the argument is NumPy and the operand is a tensor.
# Only the coerced dtype survives tracing, which is why the fed dtype is the wrong thing to name
# in a signature.
import numpy as np
import tensorflow as tf

W = tf.Variable(np.float32(0.3), name="weight")
V = tf.Variable(np.float64(0.7), name="wide_weight")


def coerced(x):
    return W * x


def declined(x):
    return W * x, V * x


def agreeing(y):
    return W * y


X = np.array([1.0, 2.0, 3.0])
assert X.dtype == np.float64

# One consumer, so the coercion is decided: the body computes in the variable's `float32`.
c = coerced(X)
assert c.dtype == tf.float32 and c.shape == (3,)

# Two consumers imposing different dtypes. Each succeeds on its own, so there is no single
# eager-effective dtype to name and the parameter keeps the dtype it is fed.
d32, d64 = declined(X)
assert d32.dtype == tf.float32 and d64.dtype == tf.float64

# The argument already carries the dtype its consumer imposes, so the coercion changes nothing.
Y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
a = agreeing(Y)
assert a.dtype == tf.float32 and a.shape == (3,)
