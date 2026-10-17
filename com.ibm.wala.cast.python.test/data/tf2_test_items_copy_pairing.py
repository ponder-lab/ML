# Test for wala/ML#1107: a dictionary copied in a loop over another dictionary's `items()` keeps
# each key paired with its own value, whether it is unpacked with `**` or read with a constant key.
import tensorflow as tf


class Emb(tf.keras.layers.Layer):

    def call(self, inputs):
        assert isinstance(inputs, tf.Tensor)
        return inputs * 2.0


class Gate(tf.keras.layers.Layer):

    def call(self, inputs):
        assert isinstance(inputs, list)
        return inputs[0] + inputs[1]


class Net(tf.keras.Model):

    def __init__(self, stimuli=None, kernel=None, **kwargs):
        super().__init__(**kwargs)
        self.stimuli = stimuli
        self.kernel = kernel

    def call(self, inputs):
        z = self.stimuli(inputs)
        return self.kernel([z, z])


def build(layers):
    built = {}
    for name, layer in layers.items():
        built[name] = layer
    return Net(**built)


net = build({"stimuli": Emb(), "kernel": Gate()})
assert net(tf.ones((2, 3))).shape == (2, 3)


def merge_dict(dict1, dict2):
    for key, value in dict2.items():
        dict1[key] = value
    return dict1


def get_dataset_size(data_file):
    assert isinstance(data_file, str)
    return len(data_file)


config = merge_dict(
    {},
    {
        "train_features_file": "src.txt",
        "learning_rate": tf.constant(0.1),
        "steps": tf.constant(3),
    },
)
assert get_dataset_size(config["train_features_file"]) == 7


def consume_pair(x):
    assert x.shape == (2,)
    return x


def copy_pairs(pairs):
    copied = {}
    for key, value in pairs:
        copied[key] = value
    return copied


pairs_copy = copy_pairs([("x", tf.ones((2,))), ("y", tf.ones((3,)))])
consume_pair(pairs_copy["x"])
