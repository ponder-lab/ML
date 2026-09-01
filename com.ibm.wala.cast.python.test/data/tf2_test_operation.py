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
import tensorflow as tf

a = tf.Variable([1.0, 2.0])


def returns_operation():
    return tf.group([a.assign([3.0, 4.0])])


def returns_no_op():
    return tf.no_op()


def returns_tensor():
    return tf.ones((2,))


# Eager: both evaluate to None. Decorating either function with `@tf.function` raises TypeError,
# "Python functions must return zero or more Tensors or ExtensionTypes or None values", which is
# the whole reason a consumer must be able to tell them apart from the tensor sibling below.
assert returns_operation() is None
assert returns_no_op() is None

assert isinstance(returns_tensor(), tf.Tensor)
assert returns_tensor().shape == (2,)
