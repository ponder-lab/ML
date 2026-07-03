# Miniature of NLPGNN's `nlpgnn/callbacks.py` (wala/ML#690): the wildcard-exported module the
# package init pulls in.
class EarlyStopping:
    def __init__(self, monitor="loss", min_delta=0):
        self.monitor = monitor
        self.min_delta = min_delta
