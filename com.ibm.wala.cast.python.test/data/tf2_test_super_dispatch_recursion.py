# Test https://github.com/wala/ML/issues/1013 family: a method re-entered on the same instance
# through a `super()` hop. The base `make_features` calls `self.make_features` on itself (a template
# step), the subclass's override calls `super().make_features`, and a parallel holder calls each
# sub-inputter's `make_features` from two sites. Every method body runs under a bounded number of
# contexts: the cycle through `super()` is recursion on one instance, not a new context per round.
import tensorflow as tf


def consume(t):
    assert t.shape == (4,)
    assert t.dtype == tf.int32
    return t


class TextIn(tf.keras.layers.Layer):
    def __init__(self, noise=True, **kwargs):
        super().__init__(**kwargs)
        self.noise = noise

    def make_features(self, element=None, features=None, training=None):
        if features is None:
            features = {}
        if "tokens" in features:
            return features
        tokens = tf.identity(element)
        if training and self.noise:
            noisy = self.make_features(features=dict(tokens=tokens), training=training)
            features["noisy_tokens"] = noisy["tokens"]
        features["tokens"] = tokens
        return features


class WordIn(TextIn):
    def make_features(self, element=None, features=None, training=None):
        features = super().make_features(
            element=element, features=features, training=training
        )
        if "ids" not in features:
            features["ids"] = tf.cast(features["tokens"], tf.int32)
        return features


class ParIn(tf.keras.layers.Layer):
    def __init__(self, inputters, **kwargs):
        super().__init__(**kwargs)
        self.inputters = inputters

    def make_features(self, element=None, features=None, training=None):
        if features is None:
            features = [None for _ in self.inputters]
        out = []
        for i, inputter in enumerate(self.inputters):
            if element is None:
                out.append(
                    inputter.make_features(features=features[i], training=training)
                )
            else:
                out.append(
                    inputter.make_features(element=element[i], training=training)
                )
        return out


par = ParIn([WordIn(), WordIn(), WordIn()])
feats = par.make_features(
    element=[tf.ones((4,)), tf.ones((4,)), tf.ones((4,))], training=True
)
consume(feats[0]["ids"])
again = par.make_features(features=feats, training=False)
consume(again[1]["ids"])
