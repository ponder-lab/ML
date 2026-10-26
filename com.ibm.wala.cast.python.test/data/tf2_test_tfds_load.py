# The elements of a dataset `tfds.load` returns are tensors, a pair of them under
# `as_supervised=True` (wala/ML#1155). TensorFlow Datasets is not installed for the test
# interpreter, so this file is not run. Under TensorFlow Datasets 4.9, `imdb_reviews/subwords8k`
# yields an `int64` text of shape `(None,)` and an `int64` label of shape `()`.
import tensorflow_datasets as tfds


def consume_supervised(inputs, outputs):
    pass


def consume_split(inputs, outputs):
    pass


def consume_info(inputs, outputs):
    pass


data = tfds.load("imdb_reviews/subwords8k", as_supervised=True)
train = data[tfds.Split.TRAIN].shuffle(1000).padded_batch(64)
for step, (inputs, outputs) in enumerate(train):
    consume_supervised(inputs, outputs)

test = tfds.load("imdb_reviews/subwords8k", split="test", as_supervised=True)
for inputs, outputs in test.batch(8):
    consume_split(inputs, outputs)

loaded, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
for inputs, outputs in loaded["train"].padded_batch(64):
    consume_info(inputs, outputs)


def consume_features(text, label):
    pass


# Without `as_supervised`, an element is a dictionary of features.
for example in tfds.load("imdb_reviews/subwords8k", split="train"):
    consume_features(example["text"], example["label"])


def consume_listed_train(inputs, outputs):
    pass


def consume_listed_test(inputs, outputs):
    pass


# A list `split` gives a list of datasets, here unpacked beside the information.
(listed_train, listed_test), listed_info = tfds.load(
    "mnist", split=["train", "test"], with_info=True, as_supervised=True
)
for inputs, outputs in listed_train.batch(8):
    consume_listed_train(inputs, outputs)
for inputs, outputs in listed_test.batch(8):
    consume_listed_test(inputs, outputs)
