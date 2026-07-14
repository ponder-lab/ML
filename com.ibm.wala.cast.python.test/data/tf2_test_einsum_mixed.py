import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def h(a):
    pass


def k(a):
    pass


def m(a):
    pass


# Ellipsis forms beyond the leading-batch idiom (wala/ML#705). Labels before an
# ellipsis bind leading axes; labels after it bind trailing ones.
t = tf.ones((2, 3, 4))
r1 = tf.einsum("i...j->i...j", t)
assert r1.shape == (2, 3, 4)
assert r1.dtype == tf.float32
f(r1)

# Implicit mode with an ellipsis: the broadcast group precedes the
# once-occurring labels, so "i...j" composes (3, 2, 4).
r2 = tf.einsum("i...j", t)
assert r2.shape == (3, 2, 4)
assert r2.dtype == tf.float32
g(r2)

# The mirror of tf2_test_einsum_batch.py's broadcast: the size-1 axis sits in
# the second input's group, so the (5,) and (2, 1) groups broadcast to (2, 5).
a = tf.ones((5, 3, 4))
b = tf.ones((2, 1, 4, 2))
r3 = tf.einsum("...ij,...jk->...ik", a, b)
assert r3.shape == (2, 5, 3, 2)
assert r3.dtype == tf.float32
h(r3)

# A statically-unknown batch axis (keras.Input's None) refines against a known
# one: the runtime requires the unknown to be 1 or equal, so the known non-1
# size wins in either input order.
c = tf.ones((2, 4))
inp = tf.keras.Input(shape=(4,))
r4 = tf.einsum("...i,...i->...", c, inp)
assert r4.shape == (2,)
assert r4.dtype == tf.float32
k(r4)

r5 = tf.einsum("...i,...i->...", inp, c)
assert r5.shape == (2,)
assert r5.dtype == tf.float32
k(r5)

# Two statically-unknown batch axes stay unknown in size but keep the rank.
r6 = tf.einsum("...i,...i->...", inp, inp)
assert r6.shape == (None,)
assert r6.dtype == tf.float32
m(r6)

# A known size-1 axis yields the other side of the broadcast, so against an
# unknown axis the result stays unknown, in either input order.
one = tf.ones((1, 4))
r7 = tf.einsum("...i,...i->...", one, inp)
assert r7.shape == (None,)
assert r7.dtype == tf.float32
m(r7)

r8 = tf.einsum("...i,...i->...", inp, one)
assert r8.shape == (None,)
assert r8.dtype == tf.float32
m(r8)
