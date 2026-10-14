# Test that a dataset built by `from_generator` with a dict of `output_types`, as a dataset class
# builds it, types the key a trainer reads from its batch inside a distributed training step.
import numpy as np
import tensorflow as tf


def trainer_sink(x):
    return x * 2.0


class AudioData:
    def get_args(self):
        return [[b"first", b"second"]]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            audio = np.ones(5 + i, dtype=np.float32)
            yield {
                "utt_ids": utt_id,
                "audios": audio,
                "audio_lengths": np.float32(5 + i),
            }

    def get_output_dtypes(self):
        return {"utt_ids": tf.string, "audios": tf.float32, "audio_lengths": tf.float32}

    def create(self, batch_size=2):
        output_types = self.get_output_dtypes()
        datasets = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, args=(self.get_args())
        )
        datasets = datasets.filter(lambda x: x["audio_lengths"] > 1)
        padded_shapes = {"utt_ids": [], "audios": [None], "audio_lengths": []}
        datasets = datasets.padded_batch(batch_size, padded_shapes=padded_shapes)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets


class Trainer:
    def __init__(self, dataset):
        self._strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        self.train_data_loader = self._strategy.experimental_distribute_dataset(dataset)

    def compute_per_example_losses(self, batch):
        audios = batch["audios"]
        return trainer_sink(tf.expand_dims(audios, -1))

    def _one_step_forward(self, batch):
        return self._strategy.run(self.compute_per_example_losses, args=(batch,))

    def train(self):
        self.one_step_forward = tf.function(
            self._one_step_forward,
            input_signature=[self.train_data_loader.element_spec],
        )
        results = []
        for steps, batch in enumerate(self.train_data_loader, 1):
            results.append(self.one_step_forward(batch))
        return results


results = Trainer(AudioData().create()).train()
assert results[0].dtype == tf.float32 and results[0].shape == (2, 6, 1)


def base_trainer_sink(x):
    return x * 2.0


class BaseTrainer:
    def __init__(self):
        self.train_data_loader = None
        self._strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    def set_train_data_loader(self, train_dataset):
        self.train_data_loader = train_dataset

    def fit(self, train_data_loader):
        self.set_train_data_loader(train_data_loader)
        self.train_data_loader = self._strategy.experimental_distribute_dataset(
            self.train_data_loader
        )
        return self.run()

    def run(self):
        results = []
        for steps, batch in enumerate(self.train_data_loader, 1):
            results.append(self._train_step(batch))
        return results

    def _train_step(self, batch):
        self.one_step_forward = tf.function(
            self._one_step_forward,
            input_signature=[self.train_data_loader.element_spec],
        )
        return self.one_step_forward(batch)

    def _one_step_forward(self, batch):
        return self._strategy.run(self.per_replica, args=(batch,))


class AudioTrainer(BaseTrainer):
    def per_replica(self, batch):
        return base_trainer_sink(tf.expand_dims(batch["audios"], -1))


results = AudioTrainer().fit(AudioData().create())
assert results[0].dtype == tf.float32 and results[0].shape == (2, 6, 1)
