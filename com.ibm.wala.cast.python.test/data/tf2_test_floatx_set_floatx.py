# Witness for wala/ML#871: when tf.keras.backend.set_floatx is reachable, the floatx() default is
# no longer statically knowable, so a dtype=floatx() token degrades to the unknown dtype rather than
# the assumed float32. set_floatx changes the backend default program-wide; assuming float32 here
# would be confidently wrong for a program that overrode it.
#
# At run time set_floatx("float64") makes floatx() return "float64", so via_floatx_call's array is
# float64. The static analysis cannot know which value set_floatx installed, so it declines to the
# unknown dtype rather than guessing; the JUnit expectation is that unknown, not this runtime dtype.
import tensorflow as tf

tf.keras.backend.set_floatx("float64")


def via_floatx_call():
    return tf.ones([2, 2], dtype=tf.keras.backend.floatx())


assert via_floatx_call().dtype == tf.float64
via_floatx_call()
