# Test that a key of the dict a `map` function returns from two sites, one storing an allocated
# tensor and one a computed value, reads either site's value.
import numpy as np
import tensorflow as tf

FROM_FILES = len("files") > 3


def sites_sink(x):
    return x


def load_audio(name):
    return np.ones(5, dtype=np.float32)


def generator():
    for i in range(2):
        yield {"audio_files": b"a", "lengths": np.int32(i + 1)}


def load(items):
    if FROM_FILES:
        return {
            "audios": tf.numpy_function(load_audio, [items["audio_files"]], tf.float32)
        }
    return {"audios": items["lengths"] * 2}


dataset = tf.data.Dataset.from_generator(
    generator, output_types={"audio_files": tf.string, "lengths": tf.int32}
)
dataset = dataset.map(load)
for element in dataset:
    result = sites_sink(element["audios"])
assert result.dtype == tf.float32
