package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ml.types.TensorType.CompoundDim;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * Tests {@link ShapeResult#widening}, the covering of one shape by another it rests on, and {@link
 * ShapeResult#intersection} (wala/ML#1293).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ShapeResultWideningTest {

  /** A size that is fixed at run time and that the analysis did not compute. */
  private static final Object U = UnresolvedDim.INSTANCE;

  /** A size that is {@code None} at run time. */
  private static final Object D = DynamicDim.INSTANCE;

  /** A ragged size. */
  private static final Object R = RaggedDim.INSTANCE;

  /**
   * Makes a shape.
   *
   * @param sizes The sizes: an {@link Integer} for a known one, or a marker.
   * @return The shape.
   */
  private static List<Dimension<?>> shape(Object... sizes) {
    List<Dimension<?>> shape = new ArrayList<>();
    for (Object size : sizes)
      shape.add(size instanceof Integer ? new NumericDim((Integer) size) : (Dimension<?>) size);
    return shape;
  }

  /**
   * Makes a result of one shape.
   *
   * @param sizes The shape's sizes, as {@link #shape} takes them.
   * @return The result.
   */
  private static ShapeResult one(Object... sizes) {
    return ShapeResult.of(Set.of(shape(sizes)));
  }

  /** A later shape that differs in one size makes that size one no one computed. */
  @Test
  public void testAxisThatDiffersBecomesUnresolved() {
    assertEquals(one(2, U, 4), one(2, 5, 4).widening(one(2, 7, 4)));
  }

  /** A later shape the result covers changes nothing, and the same result is returned. */
  @Test
  public void testCoveredShapeChangesNothing() {
    ShapeResult widened = one(2, U, 4);
    assertSame(widened, widened.widening(one(2, 9, 4)));
    assertSame(widened, widened.widening(one(2, U, 4)));

    ShapeResult dynamic = one(D, 3);
    assertSame(dynamic, dynamic.widening(one(4, 3)));
    assertSame(dynamic, dynamic.widening(one(U, 3)));
  }

  /**
   * A size that is {@code None} at run time in either shape stays so, and a ragged one stays ragged
   * (the wala/ML#721 taint), while the other sizes are widened on their own.
   */
  @Test
  public void testMarkerOfEitherShapeIsKept() {
    assertEquals(one(D, 3), one(4, 3).widening(one(D, 3)));
    assertEquals(one(D, U), one(4, 5).widening(one(D, 6)));
    assertEquals(one(D, 3), one(U, 3).widening(one(D, 3)));
    assertEquals(one(R, 3), one(4, 3).widening(one(R, 3)));
    assertEquals(one(R, 3), one(D, 3).widening(one(R, 3)));
    assertEquals(one(R, 3), one(R, 3).widening(one(D, 3)));
  }

  /**
   * A product with a factor that is {@code None} at run time, or ragged, is widened as that factor
   * is (the wala/ML#721 taint), and a size no one computed does not cover it.
   */
  @Test
  public void testMarkerInsideProductIsKept() {
    Object dynamic = new CompoundDim(shape(D, 4));
    assertEquals(one(2, D), one(2, dynamic).widening(one(2, 8)));
    assertEquals(one(2, D), one(2, 8).widening(one(2, dynamic)));
    assertEquals(one(2, R), one(2, new CompoundDim(shape(R, 4))).widening(one(2, 8)));
    assertEquals(one(2, R), one(2, new CompoundDim(shape(R, 4))).widening(one(2, dynamic)));
    assertEquals(one(2, U), one(2, new CompoundDim(shape(3, 4))).widening(one(2, 8)));

    assertFalse(ShapeResult.covers(shape(2, U), shape(2, dynamic)));
    assertTrue(ShapeResult.covers(shape(2, D), shape(2, dynamic)));
    assertTrue(ShapeResult.covers(shape(2, dynamic), shape(2, dynamic)));
  }

  /**
   * A later shape of a rank the result does not hold is taken beside its shapes, and the whole
   * shape becomes unknown where the result would then hold shapes of a third rank.
   */
  @Test
  public void testThirdRankBecomesUnknown() {
    ShapeResult ranks = ShapeResult.of(Set.of(shape(2, 3), shape(2, 2, 3)));
    assertEquals(ranks, one(2, 3).widening(one(2, 2, 3)));
    assertEquals(ShapeResult.unknown(), ranks.widening(one(3)));
    assertEquals(
        ShapeResult.unknown(),
        one(3).widening(ShapeResult.of(Set.of(shape(2, 3), shape(2, 2, 3)))));

    // A shape of a rank the result holds is widened into that rank's member, whatever else it
    // holds.
    assertEquals(ShapeResult.of(Set.of(shape(U, 3), shape(2, 2, 3))), ranks.widening(one(4, 3)));
    ShapeResult three = ShapeResult.of(Set.of(shape(3), shape(2, 3), shape(2, 2, 3)));
    assertEquals(
        ShapeResult.of(Set.of(shape(U), shape(2, 3), shape(2, 2, 3))), three.widening(one(5)));
    assertEquals(ShapeResult.unknown(), three.widening(one(2, 2, 2, 3)));
  }

  /**
   * The members of the later shape's rank become one member with it, and those of other ranks stay.
   */
  @Test
  public void testMembersOfOneRankBecomeOne() {
    ShapeResult held = ShapeResult.of(Set.of(shape(2, 3), shape(5, 3), shape(9)));
    assertEquals(ShapeResult.of(Set.of(shape(U, 3), shape(9))), held.widening(one(7, 3)));
  }

  /** A later result's shapes are each taken in turn. */
  @Test
  public void testEachLaterShapeIsTaken() {
    ShapeResult later = ShapeResult.of(Set.of(shape(2, 5, 4), shape(2, 7, 4), shape(2, 7, 6)));
    assertEquals(one(2, U, U), one(2, 5, 4).widening(later));
  }

  /** A member the result covers is dropped, also when the later result adds nothing. */
  @Test
  public void testCoveredMemberIsDropped() {
    ShapeResult held = ShapeResult.of(Set.of(shape(4, 3), shape(D, 3)));
    assertEquals(one(D, 3), held.widening(one(6, 3)));
    assertEquals(one(D, 3), held.widening(one(D, 3)));
  }

  /** A result with the unknown remainder covers every shape and is returned as it is. */
  @Test
  public void testRemainderIsKept() {
    ShapeResult unknown = ShapeResult.unknown();
    assertSame(unknown, unknown.widening(one(2, 3)));
    ShapeResult partial = new ShapeResult(Set.of(shape(2, 3)), true);
    assertSame(partial, partial.widening(one(4, 3)));

    // A later result's remainder is taken, with its members widened as any others.
    assertEquals(
        new ShapeResult(Set.of(shape(U, 3)), true),
        one(2, 3).widening(new ShapeResult(Set.of(shape(4, 3)), true)));
  }

  /**
   * The result that is not a tensor widens to the later one, and a later one of it adds nothing.
   */
  @Test
  public void testBottomIsNeutral() {
    ShapeResult held = one(2, 3);
    assertSame(held, held.widening(ShapeResult.bottom()));
    assertSame(held, ShapeResult.bottom().widening(held));
  }

  /** A shape covers another of its rank whose sizes are its own or less general. */
  @Test
  public void testCovers() {
    assertTrue(ShapeResult.covers(shape(2, U, 4), shape(2, 5, 4)));
    assertTrue(ShapeResult.covers(shape(2, 5, 4), shape(2, 5, 4)));
    assertFalse(ShapeResult.covers(shape(2, 5, 4), shape(2, U, 4)));
    assertFalse(ShapeResult.covers(shape(2, U, 4), shape(3, 5, 4)));
    assertFalse(ShapeResult.covers(shape(U, 4), shape(2, 5, 4)));

    assertTrue(ShapeResult.covers(shape(D), shape(U)));
    assertFalse(ShapeResult.covers(shape(U), shape(D)));
    assertTrue(ShapeResult.covers(shape(R), shape(D)));
    assertFalse(ShapeResult.covers(shape(D), shape(R)));
  }

  /** Members that another member covers are dropped, and members that none covers are kept. */
  @Test
  public void testWithoutCovered() {
    ShapeResult result = ShapeResult.of(Set.of(shape(4, 3), shape(D, 3), shape(7), shape(8)));
    assertEquals(ShapeResult.of(Set.of(shape(D, 3), shape(7), shape(8))), result.withoutCovered());

    ShapeResult partial = new ShapeResult(Set.of(shape(4, 3), shape(U, 3)), true);
    assertEquals(new ShapeResult(Set.of(shape(U, 3)), true), partial.withoutCovered());

    ShapeResult apart = ShapeResult.of(Set.of(shape(2, 3), shape(5, 3)));
    assertSame(apart, apart.withoutCovered());
  }

  /** Only the given shapes stand for others, and only where the result holds them. */
  @Test
  public void testWithoutCoveredBy() {
    ShapeResult result = ShapeResult.of(Set.of(shape(2, 5, 4), shape(2, U, 4), shape(4, 3)));
    assertEquals(
        ShapeResult.of(Set.of(shape(2, U, 4), shape(4, 3))),
        result.withoutCoveredBy(Set.of(shape(2, U, 4), shape(D, 3))));
    assertSame(result, result.withoutCoveredBy(Set.of()));
    assertSame(result, result.withoutCoveredBy(Set.of(shape(D, 3))));

    // A member that covers another is not one of the given shapes, so both stay.
    ShapeResult dynamic = ShapeResult.of(Set.of(shape(4, 3), shape(D, 3)));
    assertSame(dynamic, dynamic.withoutCoveredBy(Set.of(shape(2, U, 4))));
  }

  /**
   * An intersection holds the shapes both results hold, and an unknown remainder only when both
   * have one.
   */
  @Test
  public void testIntersection() {
    ShapeResult both = ShapeResult.of(Set.of(shape(2), shape(3)));
    assertSame(both, both.intersection(ShapeResult.of(Set.of(shape(2), shape(3)))));
    assertEquals(one(2), both.intersection(one(2)));
    assertEquals(ShapeResult.bottom(), one(2).intersection(one(3)));
    ShapeResult open = one(2).union(ShapeResult.unknown());
    assertEquals(one(2), open.intersection(both));
    assertEquals(one(2), both.intersection(open));
    assertEquals(open, open.intersection(both.union(ShapeResult.unknown())));
  }
}
