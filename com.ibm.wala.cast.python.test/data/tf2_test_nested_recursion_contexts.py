# A helper that recurses through its own comprehensions, reached through two wrappers that each pass
# their parameter on, so the wrappers' calls separate the helper's callers while its recursive calls
# keep one site (wala/ML#1305).
import tensorflow as tf


def map_values(config, fn):
    if isinstance(config, dict):
        return {key: map_values(value, fn) for key, value in config.items()}
    elif isinstance(config, list):
        return [map_values(elem, fn) for elem in config]
    else:
        return fn(config)


def identity_values(config):
    def _fn(value):
        return tf.identity(value)

    return map_values(config, _fn)


def doubled_values(config):
    def _fn(value):
        return value * 2

    return map_values(config, _fn)


def consume_identity(t):
    assert t.shape == (2, 3)


def consume_doubled(t):
    assert t.shape == (4,)


identity = identity_values({"a": {"b": tf.ones([2, 3])}, "c": [tf.ones([5])]})
consume_identity(identity["a"]["b"])
doubled = doubled_values({"d": [{"e": tf.ones([4])}]})
consume_doubled(doubled["d"][0]["e"])
