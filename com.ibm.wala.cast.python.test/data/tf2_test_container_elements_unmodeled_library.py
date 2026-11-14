# Test `ContainerElements` where a list is passed to a TensorFlow function no summary models,
# `tf.strings.join`: the list reports no length, since without a summary the analysis cannot tell
# that the function leaves it as it is (wala/ML#1182).
import tensorflow as tf


def sink_parts(c):
    assert len(c) == 2


parts = [tf.constant("a"), tf.constant("b")]
joined = tf.strings.join(parts, separator="-")
assert joined.numpy() == b"a-b"
sink_parts(parts)
