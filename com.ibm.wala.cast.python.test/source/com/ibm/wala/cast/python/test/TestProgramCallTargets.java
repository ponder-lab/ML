package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.util.CallTargets;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Tests {@link CallTargets} on a program that imports no library, so every answer comes from the
 * analysis of Python itself. Each {@code call_*} function of the fixture makes one call of
 * interest.
 */
public class TestProgramCallTargets extends TestJythonCallGraphShape {

  private static final String SCRIPT = "call_targets.py";

  private static PropagationCallGraphBuilder builder;

  private static CallGraph callGraph;

  @BeforeClass
  public static void analyze() throws Exception {
    PythonAnalysisEngine<?> engine = new TestProgramCallTargets().makeEngine(SCRIPT);
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
    Map<SSAAbstractInvokeInstruction, CGNode> calls = callsIn(function);
    for (SSAAbstractInvokeInstruction call : calls.keySet())
      for (CGNode target : CallTargets.programTargets(builder, calls.get(call), call))
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

  /** A callee that is one of two program functions: both. */
  @Test
  public void testEitherProgramFunction() {
    assertEquals(Set.of("helper", "other"), programTargetsIn("call_either"));
    assertEquals(Map.of("helper", false, "other", false), boundTargetsIn("call_either"));
  }

  /** A built-in function runs no program function. */
  @Test
  public void testBuiltin() {
    assertEquals(Set.of(), programTargetsIn("call_builtin"));
    assertEquals(Map.of(), boundTargetsIn("call_builtin"));
  }

  /**
   * {@link CallTargets#isProgramCode(CGNode)}: a program function's node is program code, and the
   * trampoline of its method is not, though the trampoline's class is named after the method.
   */
  @Test
  public void testIsProgramCode() {
    boolean program = false;
    boolean trampoline = false;
    for (CGNode node : callGraph) {
      if (!name(node).equals("C/m")) continue;
      if (CallTargets.isTrampoline(callGraph.getClassHierarchy(), node.getMethod())) {
        trampoline = true;
        assertFalse("A trampoline is not program code.", CallTargets.isProgramCode(node));
      } else {
        program = true;
        assertTrue("A method's own node is program code.", CallTargets.isProgramCode(node));
      }
    }
    assertTrue("`C/m` must have a node of its own.", program);
    assertTrue("`C/m` must have a trampoline's node.", trampoline);
  }
}
