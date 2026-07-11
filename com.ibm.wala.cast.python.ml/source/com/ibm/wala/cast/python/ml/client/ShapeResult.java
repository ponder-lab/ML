package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * The result of a shape resolution: the resolvable shape members plus whether an unresolvable
 * remainder exists (wala/ML#718). The legacy convention represents ⊤ as {@code null} and ⊥ as an
 * empty set, which cannot express a partially resolved value set (some members concrete, some
 * unknown), so a set read over, e.g., a loop-carried φ collapses entirely on one unknown member.
 * This record makes the unknown remainder explicit, mirroring how dtype sets already carry {@code
 * DType.UNKNOWN} in band.
 *
 * @param members The resolvable shape members. Never {@code null}; possibly empty.
 * @param hasUnknown Whether an unresolvable remainder exists alongside {@code members}.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public record ShapeResult(Set<List<Dimension<?>>> members, boolean hasUnknown) {

  /** The fully unknown result (⊤): no resolvable members and an unknown remainder. */
  private static final ShapeResult UNKNOWN = new ShapeResult(Collections.emptySet(), true);

  /** The not-a-tensor result (⊥): no members and no unknown remainder. */
  private static final ShapeResult BOTTOM = new ShapeResult(Collections.emptySet(), false);

  /**
   * Wraps fully resolved members.
   *
   * @param members The resolved shape members.
   * @return The fully resolved result.
   */
  public static ShapeResult of(Set<List<Dimension<?>>> members) {
    return new ShapeResult(members, false);
  }

  /**
   * Returns the fully unknown result (⊤).
   *
   * @return The fully unknown result.
   */
  public static ShapeResult unknown() {
    return UNKNOWN;
  }

  /**
   * Returns the not-a-tensor result (⊥).
   *
   * @return The not-a-tensor result.
   */
  public static ShapeResult bottom() {
    return BOTTOM;
  }

  /**
   * Lifts a legacy shape set into a result: {@code null} lifts to ⊤, an empty set to ⊥, and any
   * other set to a fully resolved result.
   *
   * @param shapes The legacy shape set, with {@code null} meaning ⊤ and empty meaning ⊥.
   * @return The lifted result.
   */
  public static ShapeResult fromLegacy(Set<List<Dimension<?>>> shapes) {
    if (shapes == null) return UNKNOWN;
    if (shapes.isEmpty()) return BOTTOM;
    return new ShapeResult(shapes, false);
  }

  /**
   * Collapses this result to the legacy convention: any unknown remainder collapses the whole
   * result to {@code null} (⊤), which over-approximates a partial result soundly; otherwise the
   * members stand, with an empty set meaning ⊥.
   *
   * @return The legacy shape set view.
   */
  public Set<List<Dimension<?>>> toLegacy() {
    if (this.hasUnknown()) return null;
    return this.members();
  }

  /**
   * Unions this result with another: members union and the unknown remainders disjoin.
   *
   * @param other The result to union with.
   * @return The union.
   */
  public ShapeResult union(ShapeResult other) {
    if (this.equals(other)) return this;
    Set<List<Dimension<?>>> combined = HashSetFactory.make(this.members());
    combined.addAll(other.members());
    return new ShapeResult(combined, this.hasUnknown() || other.hasUnknown());
  }

  /**
   * Decides whether this result is ⊥ (not a tensor).
   *
   * @return {@code true} iff there are no members and no unknown remainder.
   */
  public boolean isBottom() {
    return this.members().isEmpty() && !this.hasUnknown();
  }

  /**
   * Decides whether this result is partial: some members resolved alongside an unknown remainder.
   *
   * @return {@code true} iff there are members and an unknown remainder.
   */
  public boolean isPartial() {
    return !this.members().isEmpty() && this.hasUnknown();
  }
}
