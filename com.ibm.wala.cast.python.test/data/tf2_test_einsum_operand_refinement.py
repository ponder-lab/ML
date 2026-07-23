import os

import tensorflow as tf


def get_shape_list(tensor):
    return tensor.shape.as_list()


# A reshape whose target is a shape-vector slice with an opaque bound
# (wala/ML#711): the analysis cannot resolve `k`, so the result's shape is
# unknown while its dtype survives.
def opaque(t, k):
    return tf.reshape(t, get_shape_list(t)[0:k])


# The three-term binding order (wala/ML#704): the first operand binds `h`
# dynamic (keras.Input's None) and `n` to 3; the second operand's known
# `h = 6` refines the dynamic binding; the unresolved third operand's term
# proves rank 2 with `n = 3` shared and `q` unconstrained.
def project(x, inp, w):
    return tf.einsum("hn,fh,nq->fq", inp, w, x)


# A multi-shape operand (the guard-phi below) is constrained too: each
# incoming member of the proven rank fills its non-numeric axes from the
# constraint's known ones (`n = 3` from the resolved second operand).
def fill(x, w):
    return tf.einsum("nq,nf->qf", x, w)


# An unresolved operand whose term carries an ellipsis proves no rank, so
# the equation constrains nothing about it: the sound bail.
def blur(x, w):
    return tf.einsum("...i,ij->...j", x, w)


# Two call sites prove disagreeing constraints (rank 2 with a trailing 3
# versus rank 3 with a trailing 5) for the same operand, so neither is
# asserted: the conflicting refinement drops.
def choose(x, a, b, flag):
    if flag:
        return tf.einsum("ij,jk->ik", x, a)
    return tf.einsum("ijk,kl->ijl", x, b)


def choose2(x, a, b, flag):
    if flag:
        return tf.einsum("ij,jk->ik", x, a)
    return tf.einsum("ijk,kl->ijl", x, b)


def f(a):
    pass


def g(a):
    pass


def h(a):
    pass


def k(a):
    pass


def dead(flag, x):
    # This branch is dead at runtime (the runtime rejects each call), but the
    # analysis still explores it: a resolved operand whose rank contradicts
    # its term, a shared label with two unequal known sizes, and a call site
    # with fewer positional tensors than the equation's terms each prove no
    # constraint, so the unresolved operand keeps its unknown shape.
    if flag:
        t = tf.ones((2, 3, 4))
        u = tf.ones((2, 3))
        v = tf.ones((4, 5))
        f(tf.einsum("ij,jk->ik", t, x))  # Rank 3 against a rank-2 term.
        f(tf.einsum("ij,ik,jl->kl", u, v, x))  # Shared `i` is 2 and 4.
        f(tf.einsum("ij,jk->ik", x))  # A single positional tensor.


x0 = tf.ones((3, 5))
x = opaque(x0, x0.shape.ndims)

inp = tf.keras.Input(shape=(3,))
w = tf.ones((2, 6))
r1 = project(x, inp, w)
assert r1.shape == (2, 5)
assert r1.dtype == tf.float32
f(r1)

phi = (
    tf.keras.Input(shape=(5,))
    if len(get_shape_list(x0)) == 2
    else tf.keras.Input(shape=(7,))
)
w2 = tf.ones((3, 2))
r2 = fill(phi, w2)
assert r2.shape == (5, 2)
assert r2.dtype == tf.float32
g(r2)

y0 = tf.ones((2, 3))
y = opaque(y0, y0.shape.ndims)
w3 = tf.ones((3, 5))
r3 = blur(y, w3)
assert r3.shape == (2, 5)
assert r3.dtype == tf.float32
h(r3)

z0 = tf.ones((2, 3))
z = opaque(z0, z0.shape.ndims)
r4 = choose(z, tf.ones((3, 4)), tf.ones((5, 6)), True)
assert r4.shape == (2, 4)
assert r4.dtype == tf.float32
k(r4)

# The flag below is opaque to the analysis (an environment read), so both of `choose2`'s
# einsum sites stay live and their disagreeing operand constraints drop (wala/ML#704); at
# run time the unset variable takes the rank-2 arm.
flag2 = os.environ.get("ARIADNE_CHOOSE2", "") == ""
z2 = opaque(z0, z0.shape.ndims)
r5 = choose2(z2, tf.ones((3, 4)), tf.ones((5, 6)), flag2)
assert r5.shape == (2, 4)
assert r5.dtype == tf.float32
f(r5)

dead(False, x)
