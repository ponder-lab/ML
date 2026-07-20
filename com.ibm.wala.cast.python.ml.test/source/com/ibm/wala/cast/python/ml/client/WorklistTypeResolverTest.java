package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;
import org.junit.After;
import org.junit.Test;

/**
 * Unit tests for the wala/ML#365 Phase 2 engine's value semantics, driven directly with synthetic
 * queries so the corners the integration tests cannot force are pinned: the legacy null-dtype
 * translation, the {@code IllegalArgumentException} fallback, and the pure-cycle promotion.
 *
 * <p>Lives in the {@code client} package (split-package, like {@link LoggablesTest}) for the
 * engine's package-private API.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class WorklistTypeResolverTest {

  @After
  public void uninstall() {
    WorklistTypeResolver.uninstall(null);
  }

  /** An acyclic chain evaluates inline and the demanded root sees its dependency's final value. */
  @Test
  public void testAcyclicChain() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    Object result = engine.demand("A", () -> engine.read("B", () -> scalar, true), true);

    assertEquals(scalar, result);
  }

  /**
   * The legacy dtype convention returns {@code null} for ⊤ and callers null-check it to run
   * fallback arms; the engine must hand the reader back {@code null}, not a normalized set.
   */
  @Test
  public void testNullDtypeSurvives() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    assertNull(engine.demand("D", () -> null, false));
  }

  /**
   * A transfer that throws {@code IllegalArgumentException} reads as the unknown value of the
   * query's kind.
   */
  @Test
  public void testIllegalArgumentReadsAsUnknown() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    Object result =
        engine.demand(
            "E",
            () -> {
              throw new IllegalArgumentException("synthetic");
            },
            true);

    assertEquals(ShapeResult.unknown(), result);
  }

  /** A dtype query's value passes through unchanged. */
  @Test
  public void testDtypeValue() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    Object result = engine.demand("F", () -> EnumSet.of(DType.FLOAT32), false);

    assertEquals(EnumSet.of(DType.FLOAT32), result);
  }

  /**
   * A dependency cycle with no external base stabilizes at the iteration bottom, which would read
   * as "not a tensor"; the promotion pass must lift it to the unknown-marked element instead.
   */
  @Test
  public void testPureCyclePromotesToUnknown() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    // Mutually recursive transfers: each reads the other and contributes nothing of its own.
    List<Supplier<Object>> transfers = new ArrayList<>();
    transfers.add(
        () -> {
          engine.read("CYCLE_B", transfers.get(1), true);
          return ShapeResult.bottom();
        });
    transfers.add(
        () -> {
          engine.read("CYCLE_A", transfers.get(0), true);
          return ShapeResult.bottom();
        });

    Object result = engine.demand("CYCLE_A", transfers.get(0), true);

    assertEquals(ShapeResult.unknown(), result);
  }

  /**
   * The wala/ML#753 join-history class in vitro: a transfer that collapsed to the unknown-marked
   * element because it consumed an interim (still-bottom) cycle read joins the mark permanently,
   * even though its recomputation against the settled value is precise. The post-fixpoint
   * canonicalization must replace the marked value with the precise recomputation.
   */
  @Test
  public void testCanonicalizationRetractsStaleCollapse() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    List<Supplier<Object>> transfers = new ArrayList<>();
    // STALE_C collapses to ⊤ while its partner is interim and is precise once it settles.
    transfers.add(
        () -> {
          ShapeResult d = (ShapeResult) engine.read("STALE_D", transfers.get(1), true);
          return d.members().isEmpty() ? ShapeResult.unknown() : d;
        });
    // STALE_D reads its partner (recording the cycle) but has its own external base.
    transfers.add(
        () -> {
          engine.read("STALE_C", transfers.get(0), true);
          return scalar;
        });

    assertEquals(scalar, engine.demand("STALE_D", transfers.get(1), true));
    assertEquals(scalar, engine.demand("STALE_C", transfers.get(0), true));
  }

  /**
   * The canonicalization recomputes with the settled state, and a non-monotone transfer may then
   * yield ⊥ where the iteration had settled a real value; the replacement must keep the settled
   * value, since the evidence that the value is a tensor stands and the pure-cycle promotion's ⊤
   * would otherwise be reclassified as "not a tensor."
   */
  @Test
  public void testCanonicalizationKeepsSettledOverBottom() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    int[] evaluations = {0};
    List<Supplier<Object>> transfers = new ArrayList<>();
    // GUARD_G contributes its base only on the first evaluation and ⊥ afterwards.
    transfers.add(
        () -> {
          engine.read("GUARD_H", transfers.get(1), true);
          return evaluations[0]++ == 0 ? scalar : ShapeResult.bottom();
        });
    transfers.add(
        () -> {
          engine.read("GUARD_G", transfers.get(0), true);
          return ShapeResult.bottom();
        });

    assertEquals(scalar, engine.demand("GUARD_G", transfers.get(0), true));
  }

  /**
   * The wala/ML#756 perturbation seed parses defensively: an unparsable value disables the knob
   * instead of aborting the analysis.
   */
  @Test
  public void testShuffleSeedParsesDefensively() {
    System.setProperty("ariadne.typeResolution.shuffleCycles", "bogus");
    try {
      assertNull(WorklistTypeResolver.parseCycleShuffleSeed());
      System.setProperty("ariadne.typeResolution.shuffleCycles", "42");
      assertEquals(Long.valueOf(42), WorklistTypeResolver.parseCycleShuffleSeed());
    } finally {
      System.clearProperty("ariadne.typeResolution.shuffleCycles");
    }
  }

  /**
   * The perturbed engine converges cycles to the same values as the unperturbed one; the seeded run
   * also exercises the shuffle's enqueue and re-enqueue arms.
   */
  @Test
  public void testShuffledCycleConverges() {
    System.setProperty("ariadne.typeResolution.shuffleCycles", "11");
    try {
      testCanonicalizationRetractsStaleCollapse();
    } finally {
      System.clearProperty("ariadne.typeResolution.shuffleCycles");
    }
  }
}
