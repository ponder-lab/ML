# Witness for wala/ML#870: tf.keras.backend.floatx() returns the default float
# dtype token, but it is a CALL result rather than a module field, so the
# dtype-argument resolver's field-identity match could not see it and the
# allocator degraded to an unknown dtype. Modeling floatx() to allocate a DType
# the resolver recognizes by node fixes the direct call, the formal-passing
# shape (add_weight's build-method callers pass floatx()), and the module-alias
# spelling real subjects import with.
import tensorflow as tf


def via_floatx_call():
    # tf.keras.backend.floatx() returns the default float dtype ('float32').
    return tf.ones([2, 2], dtype=tf.keras.backend.floatx())


def via_field():
    # Control: the module-field spelling that DOES resolve.
    return tf.ones([2, 2], dtype=tf.float32)


via_floatx_call()
via_field()


def allocate(d):
    # The formal-passing shape: a helper receives floatx() as a formal `d` and
    # allocates with it. Mirrors add_weight's build-method callers passing floatx().
    return tf.ones([2, 2], dtype=d)


def via_formal():
    return allocate(tf.keras.backend.floatx())


def via_floatx_alias():
    # floatx reached through the module alias, as real subjects import it.
    K = tf.keras.backend
    return tf.ones([2, 2], dtype=K.floatx())


via_formal()
via_floatx_alias()
