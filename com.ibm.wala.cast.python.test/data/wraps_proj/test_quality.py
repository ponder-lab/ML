# Test for a function decorated through `functools.wraps`: the call reaches the wrapper, and the
# wrapper reaches the decorated function's body.
import tensorflow as tf

from core.quality import gaussian_noise


def test_gaussian_noise():
    image = tf.ones((4, 4, 3), dtype=tf.uint8)
    noisy = gaussian_noise(image)
    assert noisy.shape == (4, 4, 3) and noisy.dtype == tf.uint8


test_gaussian_noise()
