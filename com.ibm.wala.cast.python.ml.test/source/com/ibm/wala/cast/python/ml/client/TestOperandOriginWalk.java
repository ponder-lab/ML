package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.util.collections.Pair;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests which classifications an {@link OperandOriginWalk} keeps and replays, over graphs of
 * operands classified as {@link ElementWiseOperation} classifies them (<a
 * href="https://github.com/wala/ML/issues/1275">wala/ML#1275</a>).
 */
public class TestOperandOriginWalk {

  /** A graph of operands, and classifications of it that count the operands classified. */
  private static final class Operands {

    /** The operands, by name. */
    private final Map<String, Pair<CGNode, Integer>> operands = new HashMap<>();

    /** The two operands of each operand that is an operator's result. */
    private final Map<String, List<String>> operators = new HashMap<>();

    /** The origins of each operand that is not an operator's result. */
    private final Map<String, Set<TensorOrigin>> leaves = new HashMap<>();

    /** How many times each operand was classified. */
    private final Map<String, Integer> classified = new HashMap<>();

    /** The operands the classifications were cut at. */
    private final Set<String> cuts = new HashSet<>();

    /** The origins each operand last read, computed or replayed. */
    private final Map<String, Set<TensorOrigin>> read = new HashMap<>();

    /**
     * Adds an operand that is an operator's result.
     *
     * @param name Its name.
     * @param left The operator's first operand.
     * @param right The operator's second operand.
     * @return These operands.
     */
    private Operands operator(String name, String left, String right) {
      this.operands.put(name, Pair.make(null, this.operands.size()));
      this.operators.put(name, List.of(left, right));
      return this;
    }

    /**
     * Adds an operand that is not an operator's result.
     *
     * @param name Its name.
     * @param origin Its origin, or {@code null} for an operand without origin evidence.
     * @return These operands.
     */
    private Operands leaf(String name, TensorOrigin origin) {
      this.operands.put(name, Pair.make(null, this.operands.size()));
      this.leaves.put(name, origin == null ? null : EnumSet.of(origin));
      return this;
    }

    /**
     * Classifies an operand as {@link ElementWiseOperation} does: an operand being classified is
     * cut, a kept classification is replayed, and any other is computed from the operator's
     * operands.
     *
     * @param walk The walk.
     * @param name The operand's name.
     * @return The operand's origins, or {@code null} when it contributes no origin evidence.
     */
    private Set<TensorOrigin> classify(OperandOriginWalk walk, String name) {
      Pair<CGNode, Integer> operand = this.operands.get(name);
      if (walk.cuts(operand)) {
        this.cuts.add(name);
        return null;
      }
      OperandOriginWalk.Kept kept = walk.kept(operand);
      if (kept != null) {
        this.read.put(name, kept.copy());
        return kept.copy();
      }
      this.classified.merge(name, 1, Integer::sum);
      walk.enter(null, operand);
      Set<TensorOrigin> ret;
      if (this.operators.containsKey(name)) {
        List<String> operator = this.operators.get(name);
        ret =
            ElementWiseOperation.getDispatchedOrigins(
                this.classify(walk, operator.get(0)), this.classify(walk, operator.get(1)));
      } else ret = this.leaves.get(name);
      walk.leave(operand, ret);
      this.read.put(name, ret);
      return ret;
    }

    /**
     * Classifies an operand as the analysis did before it kept any classification: an operand being
     * classified is cut, and any other is computed again at every visit.
     *
     * @param path The operands being classified.
     * @param name The operand's name.
     * @return The operand's origins, or {@code null} when it contributes no origin evidence.
     */
    private Set<TensorOrigin> classifyPerPath(Set<String> path, String name) {
      if (!path.add(name)) return null;
      try {
        if (!this.operators.containsKey(name)) return this.leaves.get(name);
        List<String> operator = this.operators.get(name);
        return ElementWiseOperation.getDispatchedOrigins(
            this.classifyPerPath(path, operator.get(0)),
            this.classifyPerPath(path, operator.get(1)));
      } finally {
        path.remove(name);
      }
    }

    /**
     * How many times an operand was classified.
     *
     * @param name The operand's name.
     * @return The count.
     */
    private int classified(String name) {
      return this.classified.getOrDefault(name, 0);
    }
  }

  /**
   * A chain of forty operators, each over the results of the two before it, with no cycle: the walk
   * classifies each operand once, and the last result reads as it does classified per path.
   */
  @Test
  public void testSharedOperandChain() {
    Operands operands =
        new Operands().leaf("v0", TensorOrigin.NUMPY).leaf("v1", TensorOrigin.TENSORFLOW);
    for (int i = 2; i < 42; i++) operands.operator("v" + i, "v" + (i - 2), "v" + (i - 1));
    OperandOriginWalk walk = new OperandOriginWalk();
    assertEquals(EnumSet.of(TensorOrigin.TENSORFLOW), operands.classify(walk, "v41"));
    for (int i = 0; i < 42; i++) assertEquals("v" + i, 1, operands.classified("v" + i));
    assertTrue(operands.cuts.isEmpty());
    assertFalse(walk.inProgress());
  }

  /**
   * A chain of numpy operands reads numpy, and an operand without origin evidence leaves its
   * operator the other operand's origins.
   */
  @Test
  public void testChainOrigins() {
    Operands operands =
        new Operands()
            .leaf("array", TensorOrigin.NUMPY)
            .leaf("scalar", null)
            .operator("scaled", "array", "scalar")
            .operator("sum", "scaled", "array")
            .operator("total", "sum", "scaled");
    assertEquals(
        EnumSet.of(TensorOrigin.NUMPY), operands.classify(new OperandOriginWalk(), "total"));
    assertEquals(1, operands.classified("scaled"));
  }

  /**
   * Twenty diamonds in a cycle: the walk classifies each operand once, where classifying per path
   * visits the last diamond's operands once for every path through the diamonds before it, and the
   * first operand reads as it does classified per path.
   */
  @Test
  public void testDiamondsInCycle() {
    Operands operands = new Operands().leaf("array", TensorOrigin.NUMPY);
    for (int i = 0; i < 20; i++)
      operands
          .operator("top" + i, "left" + i, "right" + i)
          .operator("left" + i, "top" + (i + 1), "array")
          .operator("right" + i, "array", "top" + (i + 1));
    operands.operator("top20", "top0", "array");
    OperandOriginWalk walk = new OperandOriginWalk();
    assertEquals(EnumSet.of(TensorOrigin.NUMPY), operands.classify(walk, "top0"));
    for (String name : operands.operands.keySet()) assertEquals(name, 1, operands.classified(name));
    assertEquals(Set.of("top0"), operands.cuts);
  }

  /**
   * A classification that was cut holds while the operand it was cut at is being classified: it is
   * replayed until that operand is left and dropped then, and the classification of the operand the
   * cycle was entered at is kept for the rest of the walk.
   */
  @Test
  public void testCutClassificationHoldsForItsCycle() {
    Pair<CGNode, Integer> outer = Pair.make(null, 0);
    Pair<CGNode, Integer> first = Pair.make(null, 1);
    Pair<CGNode, Integer> member = Pair.make(null, 2);
    OperandOriginWalk walk = new OperandOriginWalk();
    walk.enter(null, outer);
    walk.enter(null, first);
    walk.enter(null, member);
    assertTrue(walk.cuts(first));
    walk.leave(member, EnumSet.of(TensorOrigin.NUMPY));
    OperandOriginWalk.Kept kept = walk.kept(member);
    assertNotNull(kept);
    assertTrue(kept.provisional());
    assertEquals(EnumSet.of(TensorOrigin.NUMPY), kept.copy());
    walk.leave(first, EnumSet.of(TensorOrigin.NUMPY));
    assertNull(walk.kept(member));
    assertNotNull(walk.kept(first));
    assertFalse(walk.kept(first).provisional());
    assertTrue(walk.inProgress());
    walk.leave(outer, EnumSet.of(TensorOrigin.NUMPY));
    assertFalse(walk.inProgress());
  }

  /**
   * A classification that replayed a cut one depends on what that one depended on, so it too is
   * dropped when the operand the cycle was entered at is left.
   */
  @Test
  public void testReplayOfCutClassificationHoldsForItsCycle() {
    Pair<CGNode, Integer> outer = Pair.make(null, 0);
    Pair<CGNode, Integer> first = Pair.make(null, 1);
    Pair<CGNode, Integer> member = Pair.make(null, 2);
    Pair<CGNode, Integer> replaying = Pair.make(null, 3);
    OperandOriginWalk walk = new OperandOriginWalk();
    walk.enter(null, outer);
    walk.enter(null, first);
    walk.enter(null, member);
    assertTrue(walk.cuts(first));
    walk.leave(member, null);
    walk.enter(null, replaying);
    assertNotNull(walk.kept(member));
    walk.leave(replaying, null);
    assertTrue(walk.kept(replaying).provisional());
    walk.leave(first, null);
    assertNull(walk.kept(replaying));
    walk.leave(outer, null);
  }

  /**
   * A classification cut in a cycle is not replayed outside it. {@code member} is first classified
   * under {@code first}, where {@code first} is cut and {@code member} reads its other operand's
   * numpy alone. Once {@code first} is left, with a parameter's origins, {@code member} is
   * classified again for {@code tail} and reads them, as {@code tail} does classified on its own.
   */
  @Test
  public void testCutClassificationIsNotReplayedOutsideItsCycle() {
    Operands operands =
        new Operands()
            .leaf("array", TensorOrigin.NUMPY)
            .leaf("parameter", TensorOrigin.PARAMETER)
            .leaf("scalar", null)
            .operator("first", "member", "parameter")
            .operator("member", "first", "array")
            .operator("tail", "member", "scalar")
            .operator("result", "first", "tail");
    assertEquals(
        EnumSet.of(TensorOrigin.PARAMETER), operands.classify(new OperandOriginWalk(), "result"));
    assertEquals(Set.of("first"), operands.cuts);
    assertEquals(EnumSet.of(TensorOrigin.PARAMETER), operands.read.get("tail"));
    assertEquals(2, operands.classified("member"));
    assertEquals(
        EnumSet.of(TensorOrigin.PARAMETER), operands.classifyPerPath(new HashSet<>(), "tail"));
  }

  /** Nothing is kept past the walk: the next one classifies every operand again. */
  @Test
  public void testNothingKeptPastTheWalk() {
    Operands operands =
        new Operands()
            .leaf("array", TensorOrigin.NUMPY)
            .operator("sum", "array", "array")
            .operator("total", "sum", "sum");
    OperandOriginWalk walk = new OperandOriginWalk();
    operands.classify(walk, "total");
    operands.classify(walk, "total");
    assertEquals(2, operands.classified("sum"));
    assertEquals(2, operands.classified("total"));
  }

  /**
   * An operand whose classification did not complete keeps nothing, and neither do the cut
   * classifications under it.
   */
  @Test
  public void testAbandonedClassification() {
    Pair<CGNode, Integer> outer = Pair.make(null, 0);
    Pair<CGNode, Integer> abandoned = Pair.make(null, 1);
    Pair<CGNode, Integer> member = Pair.make(null, 2);
    OperandOriginWalk walk = new OperandOriginWalk();
    walk.enter(null, outer);
    walk.enter(null, abandoned);
    walk.enter(null, member);
    assertTrue(walk.cuts(abandoned));
    walk.leave(member, EnumSet.of(TensorOrigin.NUMPY));
    walk.abandon(abandoned);
    assertNull(walk.kept(abandoned));
    assertNull(walk.kept(member));
    assertTrue(walk.inProgress());
    walk.leave(outer, null);
    assertFalse(walk.inProgress());
  }

  /**
   * An operand whose classification did not complete passes what it was cut at to the operand above
   * it, whose classification, completed without it, holds only until the operand of the cut is
   * left.
   */
  @Test
  public void testAbandonedClassificationPassesItsCut() {
    Pair<CGNode, Integer> outer = Pair.make(null, 0);
    Pair<CGNode, Integer> first = Pair.make(null, 1);
    Pair<CGNode, Integer> above = Pair.make(null, 2);
    Pair<CGNode, Integer> abandoned = Pair.make(null, 3);
    OperandOriginWalk walk = new OperandOriginWalk();
    walk.enter(null, outer);
    walk.enter(null, first);
    walk.enter(null, above);
    walk.enter(null, abandoned);
    assertTrue(walk.cuts(first));
    walk.abandon(abandoned);
    walk.leave(above, EnumSet.of(TensorOrigin.NUMPY));
    assertTrue(walk.kept(above).provisional());
    walk.leave(first, EnumSet.of(TensorOrigin.NUMPY));
    assertNull(walk.kept(above));
    assertNotNull(walk.kept(first));
    walk.leave(outer, null);
  }

  /**
   * Where a replay changes a classification. An operator whose operands both contribute no evidence
   * reads TensorFlow, and a cut is no evidence. Classified per path, {@code inner} asked after
   * {@code outer} classifies {@code outer} again under itself, where {@code outer}'s operands are a
   * scalar and the cut {@code inner}, so {@code outer} reads TensorFlow there and {@code result}
   * reads TensorFlow. The walk has kept {@code outer}'s classification, a parameter's, and replays
   * it, so {@code result} reads a parameter's origins, and no operand here is TensorFlow's.
   */
  @Test
  public void testReplayKeepsTheFirstClassification() {
    Operands operands =
        new Operands()
            .leaf("scalar", null)
            .leaf("parameter", TensorOrigin.PARAMETER)
            .operator("outer", "scalar", "inner")
            .operator("inner", "parameter", "outer")
            .operator("result", "outer", "inner");
    assertEquals(
        EnumSet.of(TensorOrigin.PARAMETER), operands.classify(new OperandOriginWalk(), "result"));
    assertEquals(
        EnumSet.of(TensorOrigin.TENSORFLOW), operands.classifyPerPath(new HashSet<>(), "result"));
  }
}
