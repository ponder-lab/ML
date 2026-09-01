# wala/ML#864: `tf.group` and `tf.no_op` produce an `Operation`, not a tensor. TensorFlow rejects an
# Operation as the return of a traced function, so a consumer deciding whether a function can be
# traced needs to identify one POSITIVELY. Absence of a tensor type does not serve, because a
# function returning a Python integer or None is equally absent.
#
# An Operation has neither dtype nor shape, so it is deliberately given no tensor type. What the
# model supplies is an allocation a consumer can recognise in the points-to set.
#
# Note the eager/graph split, which is why the asserts below are shaped as they are: OUTSIDE a
# trace these calls evaluate to None, and only inside one do they evaluate to an Operation. The
# analysis models the traced meaning, since that is the context whose validity a consumer is
# deciding.
import sys

import tensorflow as tf

a = tf.Variable([1.0, 2.0])


def returns_operation():
    return tf.group([a.assign([3.0, 4.0])])


def returns_no_op():
    return tf.no_op()


def returns_literal():
    # Not a tensor. Its return resolves to a non-allocation member, which the test helper records
    # rather than drops.
    return 1.0


def returns_tensor():
    return tf.ones((2,))


# Eager: both evaluate to None. Decorating either function with `@tf.function` raises TypeError,
# "Python functions must return zero or more Tensors or ExtensionTypes or None values", which is
# the whole reason a consumer must be able to tell them apart from the tensor sibling below.
assert returns_operation() is None
assert returns_no_op() is None

assert isinstance(returns_tensor(), tf.Tensor)
assert returns_tensor().shape == (2,)


def returns_print():
    return tf.print("x")


def returns_assert():
    return tf.assert_equal(1, 1)


def returns_print_kwarg():
    # The spelling real code actually uses, with a keyword the summary does not name. This is
    # COVERAGE of what a consumer will meet rather than a witness for the summary's parameter list:
    # measured both ways, it resolves to the operation with the list enumerated and with it
    # variadic, because a `<new>`+`<return>` body reads no parameter.
    return tf.print("x:", 1, output_stream=sys.stdout)


def returns_group_kwarg():
    return tf.group([], name="g")


def returns_assert_named():
    # The keyword spelling. Coverage rather than a witness, for the reason given above.
    return tf.assert_equal(1, 1, name="eq")


# The conditional pair. Eagerly these evaluate to None, exactly as `tf.group` does; only under
# tracing do they evaluate to an Operation. Decorating a function that returns either raises, which
# is why the model states the traced reading rather than the eager one.
assert returns_print() is None
assert returns_assert() is None
assert returns_assert_named() is None
assert returns_print_kwarg() is None
assert returns_group_kwarg() is None
assert returns_literal() == 1.0
