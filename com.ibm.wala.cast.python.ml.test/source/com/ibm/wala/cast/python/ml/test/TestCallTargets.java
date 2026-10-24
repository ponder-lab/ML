package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.CallTargets;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import java.io.File;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Tests {@link CallTargets}, the client-facing authority on which program functions a call runs.
 * Each {@code call_*} function of the fixture makes one call of interest.
 */
public class TestCallTargets extends TestPythonMLCallGraphShape {

  private static final String SCRIPT = "tf2_test_call_targets.py";

  private static PythonSSAPropagationCallGraphBuilder builder;

  private static CallGraph callGraph;

  @BeforeClass
  public static void analyze() throws Exception {
    PythonTensorAnalysisEngine engine =
        new TestCallTargets().makeEngine(Collections.<File>emptyList(), SCRIPT);
    builder = engine.defaultCallGraphBuilder();
    callGraph = builder.makeCallGraph(builder.getOptions());
  }

  /**
   * A node's function, named by its path in the script, such as {@code C/m}.
   *
   * @param node The node.
   * @return The name.
   */
  private static String name(CGNode node) {
    String name = node.getMethod().getDeclaringClass().getName().toString();
    int script = name.indexOf(SCRIPT + "/");
    return script < 0 ? name : name.substring(script + SCRIPT.length() + 1);
  }

  /**
   * The program functions the calls of a fixture function run, by name.
   *
   * @param function The fixture function's name.
   * @return The names.
   */
  private static Set<String> programTargetsIn(String function) {
    Set<String> ret = new TreeSet<>();
    for (SSAAbstractInvokeInstruction call : callsIn(function).keySet())
      for (CGNode target : CallTargets.programTargets(builder, callsIn(function).get(call), call))
        ret.add(name(target));
    return ret;
  }

  /**
   * The functions whose parameters receive the calls' arguments, by name, each with whether a
   * receiver is bound first.
   *
   * @param function The fixture function's name.
   * @return The names and bindings.
   */
  private static Map<String, Boolean> boundTargetsIn(String function) {
    Map<String, Boolean> ret = new TreeMap<>();
    Map<SSAAbstractInvokeInstruction, CGNode> calls = callsIn(function);
    for (SSAAbstractInvokeInstruction call : calls.keySet())
      CallTargets.parameterBoundTargets(builder, calls.get(call), call)
          .forEach((target, bound) -> ret.put(name(target), bound));
    return ret;
  }

  /** The calls in a fixture function's nodes, each with its node. */
  private static Map<SSAAbstractInvokeInstruction, CGNode> callsIn(String function) {
    Map<SSAAbstractInvokeInstruction, CGNode> ret = new LinkedHashMap<>();
    for (CGNode node : callGraph) {
      // Only program code: an instance-method trampoline is named like the method it forwards to.
      if (!CallTargets.isProgramCode(node) || !name(node).equals(function) || node.getIR() == null)
        continue;
      for (SSAInstruction instruction : node.getIR().getInstructions())
        if (instruction instanceof SSAAbstractInvokeInstruction call) ret.put(call, node);
    }
    assertTrue("`" + function + "` must be in the call graph.", !ret.isEmpty());
    return ret;
  }

  /** A direct call of a program function. */
  @Test
  public void testDirectCall() {
    assertEquals(Set.of("helper"), programTargetsIn("call_direct"));
    assertEquals(Map.of("helper", false), boundTargetsIn("call_direct"));
  }

  /** An instance method, {@code obj.m(x)}, through its trampoline, which binds the instance. */
  @Test
  public void testInstanceMethod() {
    assertEquals(Set.of("C/m"), programTargetsIn("call_method"));
    assertEquals(Map.of("C/m", true), boundTargetsIn("call_method"));
  }

  /** A bound method saved and called, {@code f = obj.m; f(x)}. */
  @Test
  public void testSavedBoundMethod() {
    assertEquals(Set.of("C/m"), programTargetsIn("call_bound"));
    assertEquals(Map.of("C/m", true), boundTargetsIn("call_bound"));
  }

  /** A static method, which binds nothing. */
  @Test
  public void testStaticMethod() {
    assertEquals(Set.of("C/s"), programTargetsIn("call_static"));
    assertEquals(Map.of("C/s", false), boundTargetsIn("call_static"));
  }

  /** A class method, through its trampoline, which binds the class. */
  @Test
  public void testClassMethod() {
    assertEquals(Set.of("C/k"), programTargetsIn("call_class"));
    assertEquals(Map.of("C/k", true), boundTargetsIn("call_class"));
  }

  /** A user Keras layer applied, {@code layer(x)}: its {@code call} runs. */
  @Test
  public void testUserLayerCall() {
    assertEquals(Set.of("MyLayer/call"), programTargetsIn("call_layer"));
    assertEquals(Map.of("MyLayer/call", true), boundTargetsIn("call_layer"));
  }

  /** A callee that is a program function or a library model: only the program function. */
  @Test
  public void testProgramFunctionBesideModel() {
    assertEquals(Set.of("helper"), programTargetsIn("call_mixed"));
    assertEquals(Map.of("helper", false), boundTargetsIn("call_mixed"));
  }

  /** A TensorFlow operation runs no program function. */
  @Test
  public void testOperation() {
    assertEquals(Set.of(), programTargetsIn("call_op"));
    assertEquals(Map.of(), boundTargetsIn("call_op"));
  }

  /** A static method called through an instance, {@code obj.s(x)}: no receiver is bound. */
  @Test
  public void testStaticMethodThroughInstance() {
    assertEquals(Set.of("C/s"), programTargetsIn("call_static_instance"));
    assertEquals(Map.of("C/s", false), boundTargetsIn("call_static_instance"));
  }

  /** A method called through {@code super()}: the instance is bound. */
  @Test
  public void testSuperMethod() {
    assertEquals(Set.of("C/m"), programTargetsIn("Sub/call_super"));
    assertEquals(Boolean.TRUE, boundTargetsIn("Sub/call_super").get("C/m"));
  }

  /**
   * A callback a library model invokes, {@code tf.map_fn(double, x)}: it runs, so it is a program
   * target, but it receives what {@code map_fn} passes, so its parameters are not bound to the
   * call's arguments.
   */
  @Test
  public void testLibraryCallback() {
    assertEquals(Set.of("double"), programTargetsIn("call_map_fn"));
    assertEquals(Map.of(), boundTargetsIn("call_map_fn"));
  }

  /**
   * {@link CallTargets#isProgramCode(CGNode)}: a program function's node is program code, and a
   * TensorFlow operation's summary node is not.
   */
  @Test
  public void testIsProgramCode() {
    boolean program = false;
    boolean model = false;
    for (CGNode node : callGraph) {
      String name = node.getMethod().getDeclaringClass().getName().toString();
      if (name(node).equals("helper")) {
        assertTrue(CallTargets.isProgramCode(node));
        program = true;
      } else if (name.equals("Ltensorflow/functions/reshape")) {
        assertFalse(CallTargets.isProgramCode(node));
        model = true;
      }
    }
    assertTrue("Both nodes must be in the call graph.", program && model);
  }
}
