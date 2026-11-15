package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.junit.Test;

/**
 * The points-to sets of values a constant guard's arm that never runs reaches: a guard an argument
 * decides (wala/ML#1187) and one a field decides (wala/ML#1256). The pointer analysis keeps both
 * arms of such a guard, so each test pins what it reads today, beside controls that read the
 * guard's operand and a shape with no dead arm.
 */
public class TestConstantGuards extends TestPythonMLCallGraphShape {

  private static final String FILE = "tf2_test_constant_guards.py";

  /** The fixture's builder, once analyzed. */
  private static PythonSSAPropagationCallGraphBuilder builder;

  /**
   * Analyzes the fixture, once.
   *
   * @return Its builder.
   */
  private static synchronized PythonSSAPropagationCallGraphBuilder builder() {
    if (builder == null)
      try {
        PythonTensorAnalysisEngine engine =
            new TestConstantGuards().makeEngine(Collections.<File>emptyList(), FILE);
        PythonSSAPropagationCallGraphBuilder b = engine.defaultCallGraphBuilder();
        b.makeCallGraph(b.getOptions());
        engine.performAnalysis(b);
        builder = b;
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }
    return builder;
  }

  /**
   * The keys a value may be, in each program body of the given name.
   *
   * @param name The body's name: the last segments of its class's name, as {@code Net/call}.
   * @param vn The value.
   * @return Per body, the keys.
   */
  private static List<Set<InstanceKey>> keys(String name, int vn) {
    PythonSSAPropagationCallGraphBuilder b = builder();
    List<Set<InstanceKey>> ret = new ArrayList<>();
    for (CGNode node : b.getCallGraph()) {
      String type = node.getMethod().getDeclaringClass().getName().toString();
      if (!type.startsWith("Lscript ") || !type.endsWith("/" + name)) continue;
      PointerKey key = b.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
      Set<InstanceKey> keys = new HashSet<>();
      b.getPointerAnalysis().getPointsToSet(key).forEach(keys::add);
      ret.add(keys);
    }
    assertFalse("Expecting " + name + " to be reached.", ret.isEmpty());
    return ret;
  }

  /**
   * What a value may be, in the one program body of the given name: a constant's value, or the name
   * of the class of any other key.
   *
   * @param name The body's name.
   * @param vn The value.
   * @return The readings.
   */
  private static Set<String> readings(String name, int vn) {
    List<Set<InstanceKey>> keys = keys(name, vn);
    assertEquals("Expecting one body named " + name + ".", 1, keys.size());
    Set<String> ret = new TreeSet<>();
    for (InstanceKey key : keys.get(0))
      ret.add(
          key instanceof ConstantKey<?> constant
              ? String.valueOf(constant.getValue())
              : key.concreteType().getName().toString());
    return ret;
  }

  /**
   * How many keys of each class a value may be, in the one program body of the given name.
   *
   * @param name The body's name.
   * @param vn The value.
   * @return Per class name, the number of keys.
   */
  private static Map<String, Integer> classCounts(String name, int vn) {
    List<Set<InstanceKey>> keys = keys(name, vn);
    assertEquals("Expecting one body named " + name + ".", 1, keys.size());
    Map<String, Integer> ret = new TreeMap<>();
    for (InstanceKey key : keys.get(0))
      ret.merge(key.concreteType().getName().toString(), 1, Integer::sum);
    return ret;
  }

  /** {@code training}'s value number in {@code Net.call(self, inputs, training, eval)}. */
  private static final int TRAINING = 4;

  /** {@code eval}'s value number in {@code Net.call(self, inputs, training, eval)}. */
  private static final int EVAL = 5;

  /** The first parameter's value number in a function, after the function itself. */
  private static final int FIRST = 2;

  /** {@code index}'s value number in a {@code message(state, index)} function. */
  private static final int INDEX = 3;

  /** The class of an {@code int} the analysis has no constant for. */
  private static final String INT = "Lint";

  /** The class of a tensor. */
  private static final String TENSOR = "Ltensorflow/python/framework/ops/Tensor";

  /**
   * A guard's operand an argument decides: {@code self.call(x, training=True)} reaches one body of
   * {@code Net.call}, where {@code training} is {@code True} and {@code eval} is {@code None}.
   */
  @Test
  public void testArgumentGuardOperands() {
    assertEquals(Set.of("true"), readings("Net/call", TRAINING));
    assertEquals(Set.of("null"), readings("Net/call", EVAL));
  }

  /**
   * The value {@code self.call(x, training=True)} returns, where only {@code if training:} runs:
   * the {@code Dense} layer's tensor. The arms that never run return a tuple and a softmax tensor
   * beside it.
   *
   * <p>TODO: https://github.com/wala/ML/issues/1187; once an arm a decided guard never takes
   * carries no value out, {@code consume} receives the one tensor, {@code Map.of(TENSOR, 1)}.
   */
  @Test
  public void testArgumentGuardResult() {
    assertEquals(Map.of("Ltuple", 1, TENSOR, 2), classCounts("consume", FIRST));
  }

  /** A guard's operand a field decides: {@code self.concat} is {@code True}, its default. */
  @Test
  public void testFieldGuardOperand() {
    assertEquals(Set.of("true"), readings("flag_full", FIRST));
    assertEquals(Set.of("true"), readings("flag_noassign", FIRST));
  }

  /**
   * The index of an {@code enumerate} over a one-element list literal, after an arm that never runs
   * assigns to the list the result of a call that subscripts it: the index is {@code 0} at run
   * time.
   *
   * <p>TODO: https://github.com/wala/ML/issues/1256; once neither route below reaches the index, it
   * is {@code Set.of("0")}.
   */
  @Test
  public void testFieldGuardBothRoutes() {
    assertEquals(Set.of("0", INT), readings("message_full", INDEX));
  }

  /**
   * As {@link #testFieldGuardBothRoutes}, but the arm only calls the function: the callee's slice
   * read is taken for a possible slice store, so the list counts as resized.
   *
   * <p>TODO: https://github.com/wala/ML/issues/1256 and https://github.com/wala/ML/issues/1130;
   * once a slice read no longer marks a list as resized, the index is {@code Set.of("0")}.
   */
  @Test
  public void testFieldGuardCallRoute() {
    assertEquals(Set.of("0", INT), readings("message_noassign", INDEX));
  }

  /**
   * As {@link #testFieldGuardBothRoutes}, but the callee does not subscript: the arm's assignment
   * reaches the loop, so the index iterates a value that is no literal.
   *
   * <p>TODO: https://github.com/wala/ML/issues/1256; once an arm a decided guard never takes
   * carries no value out, the index is {@code Set.of("0")}.
   */
  @Test
  public void testFieldGuardMergeRoute() {
    assertEquals(Set.of("0", INT), readings("message_nosub", INDEX));
  }

  /**
   * Controls: an arm that only calls a function that does not subscript, and a class with no such
   * arm. The index is {@code 0}, so a dead arm's call by itself costs nothing.
   */
  @Test
  public void testFieldGuardControls() {
    assertEquals(Set.of("0"), readings("message_bind", INDEX));
    assertEquals(Set.of("0"), readings("message_none", INDEX));
  }
}
