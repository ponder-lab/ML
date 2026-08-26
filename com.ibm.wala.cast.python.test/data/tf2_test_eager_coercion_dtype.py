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


def disagreeing(x):
    return W * x, V * x


def unaccounted(x):
    return W * x, tf.tensordot(x, V, 0)


def keyword_unaccounted(x):
    return W * x, tf.tensordot(x, b=V, axes=0)


def circular(x, y):
    return W * x, x * y


def agreeing(y):
    return W * y


def chained_inner(y):
    return W * y


def chained(x):
    return W * x, chained_inner(x)


def shaped(x):
    x.set_shape((3,))
    return W * x


X = np.array([1.0, 2.0, 3.0])
assert X.dtype == np.float64

# One consumer, so the coercion is decided: the body computes in the variable's `float32`.
c = coerced(X)
assert c.dtype == tf.float32 and c.shape == (3,)

# Two consumers imposing different dtypes. Each succeeds on its own, so there is no single
# eager-effective dtype: the body computes with `float32` at one op and `float64` at the other,
# and the union of the two is the honest answer (wala/ML#829).
d32, d64 = disagreeing(X)
assert d32.dtype == tf.float32 and d64.dtype == tf.float64

# One consumer imposes `float32`, but the other (`tf.tensordot`) declares no coercion, so the
# collection is incomplete and the parameter keeps the dtype it is fed.
u32, ut = unaccounted(X)
assert u32.dtype == tf.float32 and ut.dtype == tf.float64

# The same incompleteness with the second operand passed by keyword: a positional-only account of
# the call's arity must not let it slip past the unaccounted decline.
k32, kt = keyword_unaccounted(X)
assert k32.dtype == tf.float32 and kt.dtype == tf.float64

# One consumer imposes `float32`, but the other's partner is itself a parameter: deciding one
# undecided dtype from another would be circular, so the account is incomplete and both parameters
# keep the dtype they are fed.
Y = np.array([4.0, 5.0, 6.0])
assert Y.dtype == np.float64
c32, cy = circular(X, Y)
assert c32.dtype == tf.float32
assert cy.dtype == np.float64

# The argument already carries the dtype its consumer imposes, so the coercion changes nothing.
Z = np.array([1.0, 2.0, 3.0], dtype=np.float32)
a = agreeing(Z)
assert a.dtype == tf.float32 and a.shape == (3,)

# The chained forwarding call: at run time `chained_inner` receives the ORIGINAL value, but the
# analysis forwards the upstream parameter's coerced state, so the inner parameter's fed side is
# contaminated by the imposed dtype and must read unresolved rather than unchanged (wala/ML#838).
ci, cin = chained(X)
assert ci.dtype == tf.float32 and ci.shape == (3,)
assert cin.dtype == tf.float32 and cin.shape == (3,)

# The `set_shape` pin owns this parameter's inflow edges inside the analysis, but the callers'
# states are the runtime feed regardless: a resolved float32 tensor feed under an equal
# imposition reads unchanged, and bare conversion is safe here (wala/ML#838).
s = shaped(tf.constant([1.0, 2.0, 3.0]))
assert s.dtype == tf.float32 and s.shape == (3,)
