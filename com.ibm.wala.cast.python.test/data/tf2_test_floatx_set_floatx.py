# Witness for wala/ML#871: when tf.keras.backend.set_floatx is reachable, floatx() may return what it
# sets as well as the float32 default. set_floatx changes the backend default program-wide, and a
# call at a module's top level is restored by no function, so its effect is the whole program.
#
# At run time set_floatx("float64") makes floatx() return "float64", so via_floatx_call's array is
# float64. The analysis is flow-insensitive, so it reads both the default and the value set; the JUnit
# expectation is that union, which contains this runtime dtype.
import tensorflow as tf

tf.keras.backend.set_floatx("float64")


def via_floatx_call():
    return tf.ones([2, 2], dtype=tf.keras.backend.floatx())


assert via_floatx_call().dtype == tf.float64
via_floatx_call()
