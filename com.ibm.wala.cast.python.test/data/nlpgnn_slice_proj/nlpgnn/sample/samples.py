# Miniature of NLPGNN's `nlpgnn/sample/samples.py` (wala/ML#678): nested closures capture
# `model` lexically and dispatch `model.predict` from a frame reached by both sibling scripts.
def sample_sequence(
    model, x, length=None, context=None, temperature=1, top_k=0, top_p=1
):
    def step(tokens):
        return model.predict(tokens)

    def body(tokens):
        return step(tokens)

    return body(x)
