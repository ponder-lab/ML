import argparse
import os
import pickle
import random
import tempfile

import numpy as np

# The windowed numpy batcher at its subject shape, copied near-verbatim: an argparse root whose
# literal defaults are the only size source, a comprehension of fixed-length windows over a random
# sample, and the trailing slice pair the training loop consumes. The file discovery in the
# constructor is a bespoke stand-in; the parameter namespace stays an attribute-read so the pad
# branch is exactly as opaque as the subject's.


def consume_raw(r):
    pass


def consume_x(x):
    pass


def consume_y(y):
    pass


class _Params:
    token_eos = 1
    pad_token = 0


par = _Params()

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=2, help="batch size", type=int)
parser.add_argument("--max_seq", default=2048, help="max length", type=int)

args = parser.parse_args()

batch_size = args.batch_size
max_seq = args.max_seq


class Data:
    def __init__(self, dir_path):
        self.files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(".pickle")
        ]
        self.file_dict = {
            "train": self.files[: int(len(self.files) * 0.8)],
            "eval": self.files[int(len(self.files) * 0.8) : int(len(self.files) * 0.9)],
            "test": self.files[int(len(self.files) * 0.9) :],
        }

    def batch(self, batch_size, length, mode="train"):

        batch_files = random.sample(self.file_dict[mode], k=batch_size)

        batch_data = [self._get_seq(file, length) for file in batch_files]
        return np.array(batch_data)  # batch_size, seq_len

    def slide_seq2seq_batch(self, batch_size, length, mode="train"):
        data = self.batch(batch_size, length + 1, mode)
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y

    def _get_seq(self, fname, max_length=None):
        with open(fname, "rb") as f:
            data = pickle.load(f)
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0, len(data) - max_length)
                data = data[start : start + max_length]
            else:
                data = np.append(data, par.token_eos)
                while len(data) < max_length:
                    data = np.append(data, par.pad_token)
        return data


data_dir = tempfile.mkdtemp()
for i in range(3):
    with open(os.path.join(data_dir, "%d.pickle" % i), "wb") as f:
        pickle.dump(list(range(3000)), f)

dataset = Data(data_dir)

for b in range(1):
    batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq)

assert batch_x.shape == (2, 2048), batch_x.shape
assert batch_y.shape == (2, 2048), batch_y.shape
consume_x(batch_x)
consume_y(batch_y)

raw = dataset.batch(batch_size, max_seq + 1)
assert raw.shape == (2, 2049), raw.shape
consume_raw(raw)
