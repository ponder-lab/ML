import argparse


def consume(value):
    pass


parser = argparse.ArgumentParser()
parser.add_argument("--data", nargs="*", default=[])
args = parser.parse_args([])

# `args.data` is opaque; a subscript-slice of it is not a tensor, and neither is an element iterated
# from that slice. `value` should not be typed as a tensor. wala/ML#656.
for value in args.data[1::2]:
    consume(value)
