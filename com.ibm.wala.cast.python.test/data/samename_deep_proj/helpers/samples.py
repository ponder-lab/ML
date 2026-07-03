# Miniature of NLPGNN's `nlpgnn/sample/samples.py` closure structure (wala/ML#678): a shared
# helper whose nested closures capture `model` lexically and dispatch `model.predict` from a
# frame reached by both sibling scripts.
def sample_sequence(model, x):
    def step(tokens):
        return model.predict(tokens)

    def body(tokens):
        return step(tokens)

    return body(x)
