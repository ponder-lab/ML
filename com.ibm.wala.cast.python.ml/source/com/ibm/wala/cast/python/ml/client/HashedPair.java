package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.util.collections.Pair;

/**
 * A {@link Pair} that computes its hash code once, when it is made, for the query keys of the
 * {@link WorklistTypeResolver} (wala/ML#1176). The resolver looks its keys up in its state, its
 * dependency graph and the sets of its walks many times over, and a {@link Pair} rehashes its
 * components, often a {@link com.ibm.wala.ipa.callgraph.CGNode} and further pairs, on every lookup.
 *
 * <p>The hash code is the one {@link Pair#hashCode()} computes, and equality is {@link
 * Pair#equals(Object)}, so a hashed pair and a plain pair of equal components are equal, hash alike
 * and land in the same bucket. The components must not change their hash codes while the pair is in
 * use, as a hashed set or map already requires of its keys.
 *
 * @param <T> The type of the first component.
 * @param <U> The type of the second component.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
final class HashedPair<T, U> extends Pair<T, U> {

  private static final long serialVersionUID = 1L;

  /** The hash code {@link Pair#hashCode()} computes for the components. */
  private final int hash;

  private HashedPair(T fst, U snd) {
    super(fst, snd);
    this.hash = super.hashCode();
  }

  /**
   * Makes a pair that computes its hash code once.
   *
   * @param <T> The type of the first component.
   * @param <U> The type of the second component.
   * @param fst The first component.
   * @param snd The second component.
   * @return The pair.
   */
  static <T, U> HashedPair<T, U> of(T fst, U snd) {
    return new HashedPair<>(fst, snd);
  }

  @Override
  public int hashCode() {
    return this.hash;
  }

  @Override
  public boolean equals(Object obj) {
    return super.equals(obj);
  }
}
