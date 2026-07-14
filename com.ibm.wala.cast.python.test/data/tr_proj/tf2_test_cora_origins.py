import numpy as np

from deep_recommenders.datasets.cora import Cora

# Driver for the `Cora` data-preparation methods from
# `LongmaoTeamTf/deep_recommenders/deep_recommenders/datasets/cora.py`, vendored verbatim under
# `deep_recommenders/datasets/`; only this driver is bespoke. The four pure-numpy methods
# (`build_graph`, `encode_labels`, and `split_labels`'s nested `_sample_mask`/`_get_labels`) are
# the Hybridize-Functions-Refactoring#774 subjects: every tensor-typed def in their bodies must
# read exactly NUMPY origins so an origin-keyed consumer can decline them as convertible data
# preparation, while their parameters read PARAMETER (wala/ML#726).
cora = Cora()
ids, features, labels = cora.load_content()
graph = cora.build_graph(ids)
encoded = cora.encode_labels(labels)
(train_labels, train_mask), (valid_labels, valid_mask), (test_labels, test_mask) = (
    cora.split_labels(labels)
)
