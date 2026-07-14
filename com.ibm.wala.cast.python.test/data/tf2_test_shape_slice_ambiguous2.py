import tensorflow as tf


def top(v):
    pass


def precise(v):
    pass


def get_shape(t):
    return t.shape.as_list()


def blowup(t, f1, f2, f3, f4):
    lo = 0 if f1 else 1 if f2 else 2
    hi = 1 if f3 else 2 if f4 else 3
    return tf.reshape(tf.ones((4,)), get_shape(t)[lo:hi])


def zero_step(t, flag):
    s = 0 if flag else 2
    return tf.reshape(tf.ones((4,)), get_shape(t)[::s])


def string_bound(t, flag):
    u = "x" if flag else 2
    return tf.reshape(tf.ones((20,)), get_shape(t)[:u])


def many(t, f1, f2, f3, f4, f5, f6, f7, f8):
    u = (
        1
        if f1
        else (
            2
            if f2
            else (
                3
                if f3
                else 4 if f4 else 5 if f5 else 6 if f6 else 7 if f7 else 8 if f8 else 9
            )
        )
    )
    return tf.reshape(tf.ones((4,)), get_shape(t)[:u])


def nonconst_lower(t):
    u = len(get_shape(t))
    return tf.reshape(tf.ones((1,)), get_shape(t)[u:])


def nonconst_upper(t):
    u = len(get_shape(t))
    return tf.reshape(tf.ones((20,)), get_shape(t)[:u])


def nonconst_step(t):
    u = len(get_shape(t))
    return tf.reshape(tf.ones((4,)), get_shape(t)[::u])


def huge(t, flag):
    u = 2 if flag else 4294967298
    return tf.reshape(tf.ones((20,)), get_shape(t)[:u])


def huge_literal(t):
    return tf.reshape(tf.ones((20,)), get_shape(t)[:4294967298])


def mixed_nonconst(t, flag):
    u = 2 if flag else t
    return tf.reshape(tf.ones((20,)), get_shape(t)[:u])


def none_or_two(t, flag):
    u = None if flag else 2
    return tf.reshape(tf.ones((20,)), get_shape(t)[:u])


def dup(t, flag):
    u = 2 if flag else 2.0
    return tf.reshape(tf.ones((5,)), get_shape(t)[1:u])


# Coverage companions of tf2_test_shape_slice_ambiguous.py (wala/ML#710). Each function
# exercises one arm of the candidate-bound enumeration; the asserts document the Python
# runtime truth for the taken branch.
t = tf.ones((4, 5, 1))

# The candidate combinations (3 lowers x 3 uppers) exceed the enumeration cap: the
# analysis reports an unknown (top) shape.
a = blowup(t, True, False, True, False)
assert a.shape == (4,)
assert a.dtype == tf.float32
top(a)

# A zero step candidate is an invalid slicing: the analysis reports an unknown shape.
b = zero_step(t, False)
assert b.shape == (4, 1)
assert b.dtype == tf.float32
top(b)

# A non-numeric constant candidate cannot be a slice bound: the analysis reports an
# unknown shape.
c = string_bound(t, False)
assert c.shape == (4, 5)
assert c.dtype == tf.float32
top(c)

# Nine candidates on a single bound exceed the per-bound cap: the analysis reports an
# unknown shape.
d = many(t, True, False, False, False, False, False, False, False)
assert d.shape == (4,)
assert d.dtype == tf.float32
top(d)

# A non-constant bound (in each of the three positions) defeats the enumeration: the
# analysis reports an unknown shape.
e = nonconst_lower(t)
assert e.shape == ()
assert e.dtype == tf.float32
top(e)

g = nonconst_upper(t)
assert g.shape == (4, 5, 1)
assert g.dtype == tf.float32
top(g)

h = nonconst_step(t)
assert h.shape == (4,)
assert h.dtype == tf.float32
top(h)

# A candidate bound outside the int range has no exact projection; truncating it would
# assert a slicing the runtime value violates, so the analysis reports an unknown shape.
i = huge(t, True)
assert i.shape == (4, 5)
assert i.dtype == tf.float32
top(i)

# The same for a literal out-of-range bound (which at runtime just clamps).
j = huge_literal(t)
assert j.shape == (4, 5, 1)
assert j.dtype == tf.float32
top(j)

# A non-constant value alongside a constant candidate defeats the enumeration: the
# analysis reports an unknown shape.
k = mixed_nonconst(t, True)
assert k.shape == (4, 5)
assert k.dtype == tf.float32
top(k)

# A propagated None alongside a numeric candidate unions both slicings: (4, 5, 1) under
# [:None] and (4, 5) under [:2].
p = none_or_two(t, False)
assert p.shape == (4, 5)
assert p.dtype == tf.float32
precise(p)

# Numerically equal int and float candidates fold to one slicing, (5,) under [1:2]: the
# front end folds the float literal onto the int, so a single candidate reaches the
# analysis.
q = dup(t, True)
assert q.shape == (5,)
assert q.dtype == tf.float32
precise(q)
