import os
import pickle
import random
import tempfile

import numpy as np

token_eos = 2
pad_token = 0


class Data:
    def __init__(self, count, length):
        self.files = []
        directory = tempfile.mkdtemp()
        for i in range(count):
            path = os.path.join(directory, "seq%d.pickle" % i)
            with open(path, "wb") as f:
                pickle.dump(list(range(i, i + length + 1)), f)
            self.files.append(path)

    def batch(self, batch_size, length):
        batch_files = random.sample(self.files, k=batch_size)
        batch_data = [self._get_seq(file, length) for file in batch_files]
        return np.array(batch_data)  # batch_size, seq_len

    def slide_seq2seq_batch(self, batch_size, length):
        data = self.batch(batch_size, length + 1)
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y

    def _get_seq(self, fname, max_length=None):
        with open(fname, "rb") as f:
            data = pickle.load(f)
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0, len(data) - max_length + 1)
                data = data[start : start + max_length]
            else:
                data = np.append(data, token_eos)
                while len(data) < max_length:
                    data = np.append(data, pad_token)
        return data
