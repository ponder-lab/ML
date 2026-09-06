# Witness for wala/ML#875 and wala/ML#886: a @click.option's default is materialized as its
# parameter's default, so an unpassed option resolves inside the body to a concrete size. This is
# the gpt-2 `train` shape, whose parameters are all @click.option and are read as tensor dimensions.
#
# The defaults are DISTINCT and NOT ascending (30, 10, 20) so that a reversed or positional
# option->parameter mapping fails the witness instead of coincidentally passing: click applies
# decorators bottom-up, so the source order of the @click.option lines is the reverse of the
# parameter order, and only a by-name match lands each default on its own parameter.
import click
import tensorflow as tf


def consume_alpha(x):
    pass


def consume_beta(x):
    pass


def consume_gamma(x):
    pass


def consume_chain(x):
    pass


def consume_supplied(x):
    pass


def consume_declined(x):
    pass


class Projector:
    def __init__(self, width):
        self.width = width

    def project(self, x):
        return tf.reshape(x, [-1, self.width])


@click.command()
@click.option("--alpha", default=30)
@click.option("--beta", default=10)
@click.option("--gamma", default=20)
def train(alpha, beta, gamma):
    a = tf.ones([alpha])
    assert a.shape == (30,)
    consume_alpha(a)

    b = tf.ones([beta])
    assert b.shape == (10,)
    consume_beta(b)

    g = tf.ones([gamma])
    assert g.shape == (20,)
    consume_gamma(g)

    # The materialized default flows through a stored-attribute chain, as it does in the subject.
    p = Projector(alpha)
    c = p.project(tf.ones([4, alpha]))
    assert c.shape == (4, 30)
    consume_chain(c)


# A value supplied at the call wins over the option default: the default never overrides a passed
# argument. Here `dim` is passed 999, so the materialized default 768 must not apply.
@click.option("--dim", default=768)
def supplied(dim):
    y = tf.ones([dim])
    assert y.shape == (999,)
    consume_supplied(y)


# A non-contiguous mix: a @click.argument (no default) sits BELOW the option, so a positional read
# of the trailing defaults would bind the option's default to the wrong parameter. The whole
# function declines (its default is not materialized), which the translation exercises for every
# function it processes (see the FINE "Declining to materialize" log). Left unpassed, such an option
# cannot satisfy the call's arity, so the function is simply not reached -- the sound no-emission
# outcome -- which is why it carries no type assertion; it stands so the decline branch is covered.
@click.command()
@click.option("--size", default=55)
@click.argument("name")
def declined(size, name):
    consume_declined(tf.ones([size]))


if __name__ == "__main__":
    supplied(999)
    train()
