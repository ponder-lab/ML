# Test https://github.com/wala/ML/issues/997: a dictionary read by `get` or `pop` with a constant
# key whose default is a variable yields the default when the key is absent.
import tensorflow as tf


def consume_get(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32
    return x


def consume_pop(x):
    assert x.shape == (4, 5) and x.dtype == tf.float32
    return x


def read(config, fallback_get, fallback_pop):
    consume_get(config.get("missing", fallback_get))
    consume_pop(config.pop("absent", fallback_pop))


read({"present": 1}, tf.ones((2, 3)), tf.ones((4, 5)))
