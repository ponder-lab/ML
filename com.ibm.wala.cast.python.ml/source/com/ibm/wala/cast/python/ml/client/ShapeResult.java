package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
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
   * The ranks a widened result may hold (wala/ML#1293). A value may hold shapes of two ranks
   * without its rank growing, as one that an array and its rows both reach does; a widening that
   * would hold more takes the rank itself to be growing.
   */
  static final int WIDENED_RANKS = 2;

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
   * Intersects this result with another: the members both hold, with an unknown remainder only when
   * both have one.
   *
   * @param other The result to intersect with.
   * @return The intersection.
   */
  public ShapeResult intersection(ShapeResult other) {
    if (this.equals(other)) return this;
    Set<List<Dimension<?>>> common = HashSetFactory.make(this.members());
    common.retainAll(other.members());
    return new ShapeResult(common, this.hasUnknown() && other.hasUnknown());
  }

  /**
   * Widens this result by a later one (wala/ML#1293), for a value that an iteration keeps making
   * new shapes for. A member of the later result that a member here covers changes nothing. One
   * that none covers is not added beside the others of its rank: they and it become one member,
   * which keeps each size they all agree on and takes a marker where they differ. Where no member
   * here has its rank, it is added, unless the result would then hold shapes of more than {@link
   * #WIDENED_RANKS} ranks. The rank itself is then growing, and the result is the unknown shape,
   * with no members: the members held until then are the ranks the growth had reached when it was
   * noticed, which says nothing of the program, and a reader that takes a partial result's members
   * would take them for all there is. A result that has the remainder covers every shape and is
   * returned as it is.
   *
   * <p>The marker is {@link UnresolvedDim}, a size that is fixed at run time and that the analysis
   * did not compute, as a size that depends on how often a loop ran is. It is {@link DynamicDim}
   * where a member has that marker there, and {@link RaggedDim} likewise (the wala/ML#721 taint),
   * and a member covers another of its rank when each of its sizes is the other's or a marker at
   * least as general.
   *
   * <p>The result covers both results and holds no member another of its members covers ({@link
   * #withoutCovered}), and widening it by a result it covers returns it. A chain of widenings ends,
   * since each one that changes anything turns a size into a more general marker, merges or drops
   * members, adds a member of a rank not held, which {@link #WIDENED_RANKS} bounds, adds the
   * remainder, or gives the unknown shape.
   *
   * @param later The later result.
   * @return The widened result.
   */
  public ShapeResult widening(ShapeResult later) {
    if (this.hasUnknown() || later.isBottom()) return this;
    if (this.isBottom()) return later;
    Set<List<Dimension<?>>> widened = HashSetFactory.make(this.members());
    for (List<Dimension<?>> member : later.members()) {
      List<List<Dimension<?>>> ranked = new ArrayList<>();
      for (List<Dimension<?>> held : widened) if (held.size() == member.size()) ranked.add(held);
      if (ranked.isEmpty()) {
        widened.add(member);
        if (widened.stream().map(List::size).distinct().count() > WIDENED_RANKS) return UNKNOWN;
        continue;
      }
      if (ranked.stream().anyMatch(held -> generalize(held, member).equals(held))) continue;
      List<Dimension<?>> general = member;
      for (List<Dimension<?>> held : ranked) general = generalize(general, held);
      widened.removeAll(ranked);
      widened.add(general);
    }
    ShapeResult result = new ShapeResult(widened, later.hasUnknown()).withoutCovered();
    return result.equals(this) ? this : result;
  }

  /**
   * Returns this result without the members another of its members covers (wala/ML#1293): a shape
   * whose sizes are those of a more general member, which stands for it.
   *
   * @return The result, or this one when no member covers another.
   */
  public ShapeResult withoutCovered() {
    return this.withoutCoveredBy(this.members());
  }

  /**
   * Returns this result without the members that one of the given shapes covers, where this result
   * holds that shape too (wala/ML#1293).
   *
   * @param generals The shapes that may stand for others.
   * @return The result, or this one when it holds none of the shapes or they cover no other member.
   */
  public ShapeResult withoutCoveredBy(Set<List<Dimension<?>>> generals) {
    Set<List<Dimension<?>>> kept = HashSetFactory.make();
    for (List<Dimension<?>> member : this.members()) {
      boolean covered = false;
      for (List<Dimension<?>> general : this.members())
        if (!general.equals(member) && generals.contains(general) && covers(general, member)) {
          covered = true;
          break;
        }
      if (!covered) kept.add(member);
    }
    return kept.size() == this.members().size() ? this : new ShapeResult(kept, this.hasUnknown());
  }

  /**
   * Decides whether a shape covers another: they have one rank, and each size of the first is the
   * other's or a marker at least as general.
   *
   * @param general A shape.
   * @param member Another shape.
   * @return Whether the first covers the second.
   */
  public static boolean covers(List<Dimension<?>> general, List<Dimension<?>> member) {
    return general.size() == member.size() && generalize(general, member).equals(general);
  }

  /**
   * Returns the shape that covers two shapes of one rank: each size both have, and a marker where
   * they differ.
   *
   * @param first A shape.
   * @param second Another shape of the same rank.
   * @return The covering shape.
   */
  private static List<Dimension<?>> generalize(
      List<Dimension<?>> first, List<Dimension<?>> second) {
    List<Dimension<?>> general = new ArrayList<>(first.size());
    for (int i = 0; i < first.size(); i++) {
      Dimension<?> one = first.get(i);
      Dimension<?> other = second.get(i);
      if (Objects.equals(one, other)) general.add(one);
      else if (one instanceof RaggedDim || other instanceof RaggedDim)
        general.add(RaggedDim.INSTANCE);
      else if (one instanceof DynamicDim || other instanceof DynamicDim)
        general.add(DynamicDim.INSTANCE);
      else general.add(UnresolvedDim.INSTANCE);
    }
    return general;
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
