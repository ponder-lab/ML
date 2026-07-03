# Driver for `TuckERLoader.target_convert` from `kyzhouhzau/NLPGNN/nlpgnn/datas/graphloader.py`
# (the wala/ML#618 row): `targets` receives a `tf.data` `padded_batch` element (int32). The
# loader is vendored verbatim from upstream; only this driver, the tiny `data/` triple files,
# and the `nlpgnn/gnn/utils.py` reachable slice are bespoke.
import tensorflow as tf

from nlpgnn.datas.graphloader import TuckERLoader

batch_size = 2

loader = TuckERLoader(base_path="data", reverse=True)
er_vocab, er_vocab_pairs = loader.data_dump("train")
dataset = loader.get_batch(er_vocab, er_vocab_pairs, batch_size=batch_size)
num_entities = len(loader.entities)

for batch in dataset:
    targets = batch["t"]
    converted = loader.target_convert(targets, batch_size, num_entities)
    assert targets.dtype == tf.int32
    assert targets.shape[0] == batch_size
    assert tuple(converted.shape) == (batch_size, num_entities)
    break
