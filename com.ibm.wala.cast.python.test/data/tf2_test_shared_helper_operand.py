# Test an elementwise operation whose operand is a call result the analysis resolves only in part:
# a `log10` helper called from one method at two sites, once on a tensor of unknown shape and once
# on a scalar, so the shared `tf.math.log` the helper calls merges both arguments. The division's
# unknown-shape operand must not be read as the scalar the other site supplied, so the features it
# feeds, expanded by one axis, read as unknown rank and not as a vector of one.
import tensorflow as tf


def consume_features(t):
    assert t.shape.rank == 3


def consume_scalar(t):
    assert t.shape == ()


def consume_known(t):
    assert t.shape == (2, 3, 1)


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


class Featurizer(object):
    def __init__(self, top_db=80.0):
        self.top_db = top_db

    def power_to_db(self, S, amin=1e-10):
        log_spec = 10.0 * log10(tf.maximum(amin, S))
        log_spec -= 10.0 * log10(tf.maximum(amin, 1.0))
        consume_scalar(log10(tf.maximum(amin, 1.0)))
        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - self.top_db)
        return log_spec

    def extract(self, signal):
        spectrogram = tf.square(tf.abs(tf.signal.rfft(signal, [512])))
        features = self.power_to_db(spectrogram)
        features = tf.expand_dims(features, axis=-1)
        consume_features(features)
        return features

    def extract_known(self, spectrogram):
        features = self.power_to_db(spectrogram)
        features = tf.expand_dims(features, axis=-1)
        consume_known(features)
        return features


featurizer = Featurizer()
featurizer.extract(tf.ones((3, 512)))
featurizer.extract_known(tf.ones((2, 3)))
