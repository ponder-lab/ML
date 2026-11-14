# Test `ContainerElements` where a list is passed to a TensorFlow function no summary models,
# `tf.strings.join`: TensorFlow is registered as a library whose functions resize no list passed to
# them, so the list keeps its length (wala/ML#1182).
import tensorflow as tf


def sink_parts(c):
    assert len(c) == 2


parts = [tf.constant("a"), tf.constant("b")]
joined = tf.strings.join(parts, separator="-")
assert joined.numpy() == b"a-b"
sink_parts(parts)
