# Test for https://github.com/wala/ML/issues/1067.
import tensorflow as tf


def to_numpy(t):
    return t.numpy()


def predict(predictions):
    return tf.nest.map_structure(to_numpy, predictions)


converted = predict({"ids": tf.constant([1, 2]), "scores": tf.constant([0.5, 0.25])})
assert converted["ids"].tolist() == [1, 2]
