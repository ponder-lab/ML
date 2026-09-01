# wala/ML#864 follow-on: reading `.op` on a variable assignment yields an `Operation`, and
# TensorFlow rejects an Operation as the return of a traced function. A consumer deciding whether a
# function can be traced needs to identify one POSITIVELY, so the assignment family has to resolve.
#
# Before this was modeled the assignment methods were absent from the model file entirely, so the
# RECEIVER of the `.op` read resolved to nothing. That is a different defect from a missing `op`
# field on a resolved receiver, and the pair of functions below separates them: `returns_assignment`
# pins that the assignment itself resolves, and `returns_assignment_op` pins that its `op` does.
#
# `.op` here is a CONDITIONAL producer, the same class as `tf.print` and `tf.assert_equal`: it is
# `None` eagerly and an Operation under tracing. Confirmed against the runtime for all six methods,
# and the asserts below pin the EAGER behaviour so the thing being approximated sits beside the
# model. The model states the traced reading because that is the context a traceability question is
# always about. The RECEIVER is unconditional by contrast: the assignment itself is a real value
# either way, which is why the first two functions are a pair.

import tensorflow as tf

# A scalar, so the minimal reproduction below can be spelled exactly as reported.
v = tf.Variable(1.0)
# The scatter methods need an indexable variable and an `IndexedSlices` delta.
w = tf.Variable([1.0, 2.0])
delta = tf.IndexedSlices(tf.constant([3.0]), tf.constant([0]))


def returns_assignment():
    # The receiver on its own. Not an operation: it is the assignment's value.
    return v.assign_add(1.0)


def returns_assignment_op():
    # The minimal reproduction, verbatim rather than paraphrased: a paraphrase of a construct is not
    # the construct, and a reduction that varies the wrong thing passes while the real case fails.
    return v.assign_add(1.0).op


def returns_assign_op():
    return v.assign(3.0).op


def returns_assign_sub_op():
    return v.assign_sub(1.0).op


def returns_scatter_add_op():
    return w.scatter_add(delta).op


def returns_scatter_sub_op():
    return w.scatter_sub(delta).op


def returns_scatter_update_op():
    return w.scatter_update(delta).op


# The receiver resolves unconditionally.
assert returns_assignment().shape == ()
# Every `.op` is None EAGERLY. Under tracing each is an Operation, which is what the model states
# and what makes decorating any of these functions raise TypeError.
assert returns_assignment_op() is None
assert returns_assign_op() is None
assert returns_assign_sub_op() is None
assert returns_scatter_add_op() is None
assert returns_scatter_sub_op() is None
assert returns_scatter_update_op() is None
