# Test that a loop over a dataset wrapped in `tqdm`, the progress bar that yields the elements of
# the iterable it wraps, types each element as the dataset's.
import tensorflow as tf
from tqdm import tqdm


def enumerated_sink(x):
    return x


def plain_sink(x):
    return x


def attribute_sink(x):
    return x


def records():
    return tf.data.Dataset.from_tensor_slices(
        {
            "audios": tf.ones((4, 3), dtype=tf.float32),
            "ids": tf.zeros((4,), dtype=tf.int32),
        }
    ).batch(2)


for steps, batch in enumerate(tqdm(records(), desc="[eval]"), 1):
    enumerated_sink(batch["audios"])
    assert batch["audios"].dtype == tf.float32 and batch["audios"].shape == (2, 3)

for batch in tqdm(records()):
    plain_sink(batch["ids"])
    assert batch["ids"].dtype == tf.int32 and batch["ids"].shape == (2,)


class Evaluator:
    def __init__(self, dataset):
        self.eval_data_loader = dataset

    def evaluate(self):
        for eval_steps, batch in enumerate(
            tqdm(self.eval_data_loader, desc="[eval]"), 1
        ):
            attribute_sink(batch["audios"])
            assert batch["audios"].dtype == tf.float32


Evaluator(records()).evaluate()
