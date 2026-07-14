import tensorflow as tf


def f(a):
    pass


def dead(flag):
    # This branch is dead at runtime (the equations below are rejected by the
    # runtime), but the analysis still types each call: every malformed or
    # unsatisfiable equation composes no shape, so each result is an unknown
    # (top) shape with a precise float32 dtype.
    if flag:
        t = tf.ones((2, 3))
        u = tf.ones((3, 2))
        v = tf.ones((2, 4))
        w = tf.ones((3, 4))
        f(tf.einsum("i.j", t))  # A dot outside an ellipsis in an input term.
        f(tf.einsum("ij.", t))  # An ellipsis truncated at the term end.
        f(tf.einsum("i..jk", t))  # A two-dot run that never completes an ellipsis.
        f(tf.einsum("...i...j", t))  # A second ellipsis within one term.
        f(tf.einsum("i1", t))  # A non-letter label.
        f(tf.einsum("ij->i.j", t))  # A malformed output term.
        f(tf.einsum("ij,jk->ii", t, u))  # A repeated output label.
        f(tf.einsum("ijk", t))  # More labels than the input's rank.
        f(tf.einsum("...ijk", t))  # Ellipsis letters exceeding the rank.
        f(tf.einsum("...i,...i->...i", v, w))  # Batch axes that don't broadcast.
        f(tf.einsum("...i,...i->i", v, v))  # Broadcast axes but no output ellipsis.


dead(False)

# An empty equation is runtime-valid on a scalar (an identity), but the parser
# treats it as unresolvable, so the analysis reports an unknown shape.
r = tf.einsum("", tf.constant(3.0))
assert r.shape == ()
assert r.dtype == tf.float32
assert float(r) == 3.0
f(r)
