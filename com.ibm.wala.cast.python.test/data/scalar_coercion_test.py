# Test for wala/ML#992: a Python scalar computed from Python numbers (`n ** 0.5`) beside a tensor
# imposes no dtype on the tensor; TensorFlow converts the scalar to the tensor's dtype. The
# parameter-dtype coercion of wala/ML#828 must therefore leave `inputs` at its fed dtype, in both
# the binary and the augmented form. The test method is the entrypoint, so the file is named as
# pytest finds it.
import tensorflow as tf


def sink(inputs):
    assert inputs.dtype == tf.float32 and inputs.shape == (4, 5, 10)
    return inputs


def scale(inputs, n):
    assert inputs.dtype == tf.float32
    scaled = inputs * n**0.5
    return sink(scaled)


def scale_in_place(inputs, n):
    assert inputs.dtype == tf.float32
    inputs *= n**0.5
    assert inputs.dtype == tf.float32
    return inputs


def scale_negated(inputs, n):
    # A unary minus over the computed number: still a Python number beside the tensor.
    assert inputs.dtype == tf.float32
    scaled = inputs * -(n**0.5)
    assert scaled.dtype == tf.float32
    return scaled


def scale_by_literal(inputs):
    assert inputs.dtype == tf.float32
    scaled = inputs * 4.5
    assert scaled.dtype == tf.float32
    return scaled


class EncoderTest(tf.test.TestCase):
    def testScaling(self):
        scale(tf.random.uniform([4, 5, 10]), 20)
        scale_in_place(tf.random.uniform([4, 5, 10]), 20)
        scale_negated(tf.random.uniform([4, 5, 10]), 20)
        scale_by_literal(tf.random.uniform([4, 5, 10]))


if __name__ == "__main__":
    tf.test.main()
