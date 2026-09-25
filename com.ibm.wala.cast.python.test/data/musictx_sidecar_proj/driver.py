import numpy as np
from data import Data


def consume_seq(s):
    assert len(s) == 4


def consume_batch(b):
    assert b.shape == (2, 4) and b.dtype == np.int64


def consume_x(x):
    assert x.shape == (2, 3) and x.dtype == np.int64


dataset = Data(2, 4)
consume_seq(dataset._get_seq(dataset.files[0], 4))
consume_batch(dataset.batch(2, 4))
batch_x, batch_y = dataset.slide_seq2seq_batch(2, 3)
consume_x(batch_x)
