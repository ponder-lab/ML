# Pytest-shaped driver (wala/ML#961 miss trace): a test-file function is an entrypoint whose parameters
# are mined, and its call to the model omits the defaulted `past`, so the trampoline binds the default.
# The question is what `past_layer` reads in the contexts this entrypoint creates: {None} (the default
# bound as the null constant) or an empty set (a fresh unknown per parameter).
import tensorflow as tf

from A import Gpt2


def test_forward_omits_past(x):
    model = Gpt2(
        num_layers=2,
        d_model=8,
        num_heads=2,
        dff=16,
        max_seq_len=8,
        vocab_size=10,
    )
    logits, presents = model(x, training=False)
    return logits
