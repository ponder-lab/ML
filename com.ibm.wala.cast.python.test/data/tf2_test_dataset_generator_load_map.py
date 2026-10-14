# Test that a dataset built by `from_generator` with a dict of file names, whose `map` loads each
# file into a new dict of arrays and a second `map` rebuilds that dict, types each key a trainer
# reads from its padded batch, and that a key no `map` return or `padded_shapes` entry names is
# unknown.
import numpy as np
import tensorflow as tf


def load_sink(x):
    return x * 2.0


def mels_sink(x):
    return x * 2.0


def local_sink(x):
    return x * 2.0


def built_sink(x):
    return x * 2.0


def rewritten_sink(x):
    return x


def load_audio(name):
    return np.ones(5 + len(name), dtype=np.float32)


def load_mel(name):
    return np.ones((2 + len(name), 80), dtype=np.float32)


def collater(items):
    audio, mel = items["audios"], items["mels"]
    return {
        "utt_ids": items["utt_ids"],
        "audios": audio,
        "mels": mel,
        "mel_lengths": len(mel),
        "audio_lengths": len(audio),
    }


class AudioMelData:
    def get_args(self):
        return [[b"a", b"bb"]]

    def generator(self, utt_ids):
        for utt_id in utt_ids:
            yield {
                "utt_ids": utt_id,
                "audio_files": utt_id,
                "mel_files": utt_id,
            }

    def get_output_dtypes(self):
        return {"utt_ids": tf.string, "audio_files": tf.string, "mel_files": tf.string}

    def _load_data(self, items):
        audio = tf.numpy_function(load_audio, [items["audio_files"]], tf.float32)
        mel = tf.numpy_function(load_mel, [items["mel_files"]], tf.float32)
        return {
            "utt_ids": items["utt_ids"],
            "audios": audio,
            "mels": mel,
            "mel_lengths": len(mel),
            "audio_lengths": len(audio),
        }

    def create(self, batch_size=2, map_fn=collater):
        output_types = self.get_output_dtypes()
        datasets = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, args=(self.get_args())
        )
        datasets = datasets.map(lambda items: self._load_data(items))
        datasets = datasets.filter(lambda x: x["audio_lengths"] > 1)
        datasets = datasets.map(map_fn)
        padded_shapes = {
            "utt_ids": [],
            "audios": [None],
            "mels": [None, 80],
            "mel_lengths": [],
            "audio_lengths": [],
        }
        datasets = datasets.padded_batch(batch_size, padded_shapes=padded_shapes)
        return datasets


class Trainer:
    def __init__(self, dataset):
        self.train_data_loader = dataset

    def compute_per_example_losses(self, batch):
        mels_sink(tf.expand_dims(batch["mels"], -1))
        return load_sink(tf.expand_dims(batch["audios"], -1))

    def train(self):
        results = []
        for steps, batch in enumerate(self.train_data_loader, 1):
            results.append(self.compute_per_example_losses(batch))
        return results


results = Trainer(AudioMelData().create()).train()
assert results[0].dtype == tf.float32 and results[0].shape == (2, 7, 1)


local_results = []
for steps, batch in enumerate(AudioMelData().create(), 1):
    local_results.append(local_sink(tf.expand_dims(batch["audios"], -1)))
assert local_results[0].dtype == tf.float32 and local_results[0].shape == (2, 7, 1)


def load_by_call(items):
    audio = tf.numpy_function(load_audio, [items["audio_files"]], tf.float32)
    return dict(utt_ids=items["utt_ids"], audios=audio, audio_lengths=len(audio))


built = tf.data.Dataset.from_generator(
    AudioMelData().generator,
    output_types=AudioMelData().get_output_dtypes(),
    args=(AudioMelData().get_args()),
)
built = built.map(load_by_call)
built = built.padded_batch(
    2, padded_shapes={"utt_ids": [], "audios": [None], "audio_lengths": []}
)
built_results = []
for steps, batch in enumerate(built, 1):
    built_results.append(built_sink(tf.expand_dims(batch["audios"], -1)))
assert built_results[0].dtype == tf.float32 and built_results[0].shape == (2, 7, 1)


def frames_generator():
    for i in range(2):
        yield {"lengths": np.int32(4), "frames": np.ones((4, 2), dtype=np.float32)}


def scale(items):
    items["frames"] = items["frames"] * 2.0
    return items


rewritten = tf.data.Dataset.from_generator(
    frames_generator,
    output_types={"lengths": tf.int32, "frames": tf.float32},
    output_shapes={"lengths": [], "frames": [4, 2]},
)
rewritten = rewritten.map(scale).batch(2)
for batch in rewritten:
    rewritten_result = rewritten_sink(batch["frames"])
assert rewritten_result.dtype == tf.float32 and rewritten_result.shape == (2, 4, 2)
