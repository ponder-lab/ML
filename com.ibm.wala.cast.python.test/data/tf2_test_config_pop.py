# Test https://github.com/wala/ML/issues/1339: a `pop` or `get` with a default, on a dictionary
# that `yaml.safe_load` returns, yields the value the file sets, which the analysis does not know.
import os

import tensorflow as tf
import yaml


class DatasetConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.use_tf = config.pop("use_tf", False)
        self.batch_size = config.pop("batch_size", 1)


class Config:
    def __init__(self, path):
        with open(path) as f:
            config = yaml.safe_load(f)
        self.dataset = DatasetConfig(config.pop("dataset", {}))
        self.scale = config.get("scale", 1.0)


def consume_flagged(t):
    assert t.shape == (4, 3)


def consume_batch(t):
    assert t.shape == (2, 3)


def main():
    config = Config(os.path.join(os.path.dirname(__file__), "tf2_test_config_pop.yml"))
    x = tf.ones([4, 3])
    if config.dataset.use_tf:
        consume_flagged(x)
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.padded_batch(
        config.dataset.batch_size, padded_shapes=[3], drop_remainder=True
    )
    for batch in ds:
        consume_batch(batch)


main()
