# A class with two bases whose methods reach the first base's chain through `super()`, the shape
# of an example inputter built on a parallel inputter plus an adapter mixin (wala/ML#1014).
import tensorflow as tf


class Inputter(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._asset_prefix = ""

    def initialize(self, data_config):
        return None

    def make_dataset(self, data_file, training=None):
        raise NotImplementedError()

    def get_dataset_size(self, data_file):
        return None

    def make_features(self, element=None, features=None, training=None):
        return element

    def call(self, features, training=None):
        return features


class WordEmbedder(Inputter):

    def make_dataset(self, data_file, training=None):
        return tf.data.Dataset.from_tensor_slices(tf.ones([4, 3]))

    def make_features(self, element=None, features=None, training=None):
        return element * 2.0


class MultiInputter(Inputter):

    def __init__(self, inputters, reducer=None):
        dtype = inputters[0].dtype
        super().__init__(dtype=dtype)
        self.inputters = inputters
        self.reducer = reducer

    def __getattribute__(self, name):
        if name == "built":
            return all(inputter.built for inputter in self.inputters)
        else:
            return super().__getattribute__(name)

    def initialize(self, data_config):
        for inputter in self.inputters:
            inputter.initialize(data_config)


class ParallelInputter(MultiInputter):

    def __init__(self, inputters, reducer=None, combine_features=True):
        super().__init__(inputters, reducer=reducer)
        self.combine_features = combine_features

    def make_dataset(self, data_file, training=None):
        datasets = [
            inputter.make_dataset(data, training=training)
            for inputter, data in zip(self.inputters, data_file)
        ]
        return tf.data.Dataset.zip(tuple(datasets))

    def get_dataset_size(self, data_file):
        sizes = [
            inputter.get_dataset_size(data)
            for inputter, data in zip(self.inputters, data_file)
        ]
        return sizes[0]

    def make_features(self, element=None, features=None, training=None):
        return [
            inputter.make_features(e, training=training)
            for inputter, e in zip(self.inputters, element)
        ]


class ExampleInputterAdapter:

    def make_training_dataset(self, data_file, batch_size):
        dataset = self.make_dataset(data_file, training=True)
        return dataset.batch(batch_size)


def consume_features(features):
    return features


class ExampleInputter(ParallelInputter, ExampleInputterAdapter):

    def __init__(self, features_inputter, labels_inputter):
        self.features_inputter = features_inputter
        self.labels_inputter = labels_inputter
        super().__init__(
            [self.features_inputter, self.labels_inputter], combine_features=False
        )

    def initialize(self, data_config):
        super().initialize(data_config)

    def make_dataset(self, data_file, training=None):
        return super().make_dataset(data_file, training=training)

    def get_dataset_size(self, data_file):
        return super().get_dataset_size(data_file)

    def make_features(self, element=None, features=None, training=None):
        features, labels = super().make_features(element, training=training)
        return consume_features(features), labels


inputter = ExampleInputter(WordEmbedder(), WordEmbedder())
inputter.initialize({})
assert inputter.get_dataset_size(["a", "b"]) is None
dataset = inputter.make_training_dataset(["a", "b"], 2)
for element in dataset:
    features, labels = inputter.make_features(element)
    assert features.shape == (2, 3)
