# The NLPGNN TUDataset downstream chain (wala/ML#796): a typed reader output must survive
# load-time slicing comprehensions with `.tolist()`, the tuple return and unpack, the sampling
# generator's tuple yield and for-loop unpack, and the merge-time `np.array` rebase with
# `tf.concat` to reach the model-call parameters.
import numpy as np
import tensorflow as tf


def consume_loaded(a):
    pass


def consume_sampled(b):
    pass


def consume_merged(c):
    pass


class Loader:
    def read_file(self, src, dtype=None):
        rows = [[dtype(x) for x in line.split(",")] for line in src]
        return np.array(rows, dtype=dtype)

    def read_data(self, src):
        edge_index = self.read_file(src, dtype=int) - 1
        batch = self.read_file(src, dtype=int) - 1
        return edge_index, batch

    def load(self, src):
        edge_index, batch = self.read_data(src)
        value = [(0, 1), (1, 2)]
        edge_index = [edge_index[se[0] : se[1]].tolist() for se in value]
        batch = [batch[se[0] : se[1]].tolist() for se in value]
        return edge_index, batch

    def sample(self, data):
        edge_index, batch = data
        for i in range(2):
            nedge_index = [edge_index[j] for j in [i]]
            nbatch = [batch[j] for j in [i]]
            yield nedge_index, nbatch


def merge(edge_index, batch):
    lis_edge_index = [np.array(item) for item in edge_index]
    lis_batch = [np.array(item) for item in batch]
    strips = [0] + [i.size for i in lis_batch]
    strips = np.cumsum(strips)
    nlis_edge_index = [w + strips[i] for i, w in enumerate(lis_edge_index)]
    edge_index = tf.concat(nlis_edge_index, 0)
    batch = tf.concat(lis_batch, 0)
    return edge_index, batch


loader = Loader()
train_data = loader.load(["1,2", "3,4", "5,6"])
consume_loaded(train_data)
for edge_index, batch in loader.sample(train_data):
    consume_sampled(edge_index)
    edge_index, batch = merge(edge_index, batch)
    assert edge_index.dtype == tf.int64
    assert batch.dtype == tf.int64
    consume_merged(edge_index)
