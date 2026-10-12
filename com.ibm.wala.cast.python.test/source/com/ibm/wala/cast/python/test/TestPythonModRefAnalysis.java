package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.modref.PythonModRef;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.StaticFieldKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.modref.ExtendedHeapModel;
import com.ibm.wala.ipa.modref.ModRef;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;

public class TestPythonModRefAnalysis extends TestJythonCallGraphShape {

  private OrdinalSet<PointerKey> getModSet(String filename)
      throws ClassHierarchyException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine(filename);
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());

    Collection<CGNode> nodes = getNodes(CG, "script " + filename + "/f");
    assertEquals(1, nodes.size());

    assertTrue(nodes.iterator().hasNext());
    CGNode fNode = nodes.iterator().next();

    ModRef<InstanceKey> modRef = new PythonModRef();
    Map<CGNode, OrdinalSet<PointerKey>> mod = modRef.computeMod(CG, builder.getPointerAnalysis());

    // what heap locations does f() (transitively) modify?
    OrdinalSet<PointerKey> modSet = mod.get(fNode);
    return modSet;
  }

  @Test
  public void testGlobalMods()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    OrdinalSet<PointerKey> modSet = getModSet("globals.py");

    // should only modify the global.
    assertEquals(1, modSet.size());

    assertTrue(modSet.iterator().hasNext());
    PointerKey pointerKey = modSet.iterator().next();

    assertTrue(pointerKey instanceof StaticFieldKey);
    StaticFieldKey staticFieldKey = (StaticFieldKey) pointerKey;

    IField field = staticFieldKey.getField();
    Atom name = field.getName();
    assertEquals(Atom.findOrCreateAsciiAtom("global a"), name);
  }

  @Test
  public void testKeywordArgs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    OrdinalSet<PointerKey> modSet = getModSet("kw_args.py");
    assertEquals("Should be no modifications to heap locations in this file.", 0, modSet.size());
  }

  /**
   * The names of the global variables among the given keys.
   *
   * @param keys The keys.
   * @return The names of the fields of the {@link StaticFieldKey}s among {@code keys}.
   */
  private static Set<String> globalNames(Iterable<PointerKey> keys) {
    Set<String> names = HashSetFactory.make();
    for (PointerKey key : keys)
      if (key instanceof StaticFieldKey)
        names.add(((StaticFieldKey) key).getField().getName().toString());
    return names;
  }

  /**
   * The node of the given function in the given call graph.
   *
   * @param cg The call graph.
   * @param function The function, as a path below the script.
   * @return The function's only node.
   */
  private CGNode getNode(CallGraph cg, String function) {
    Collection<CGNode> nodes = getNodes(cg, "script globals_two_writers.py/" + function);
    assertEquals(1, nodes.size());
    return nodes.iterator().next();
  }

  /**
   * Computes the mod sets of {@code globals_two_writers.py} and checks that its global writes find
   * their keys from one read of the analysis's pointer keys, and that each function's mod set holds
   * the globals it writes (https://github.com/wala/ML/issues/1063).
   *
   * @param modRef The mod-ref analysis.
   */
  private void assertGlobalWritesReadPointerKeysOnce(ModRef<InstanceKey> modRef)
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("globals_two_writers.py");
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();

    AtomicInteger reads = new AtomicInteger();
    @SuppressWarnings("unchecked")
    PointerAnalysis<InstanceKey> counting =
        (PointerAnalysis<InstanceKey>)
            Proxy.newProxyInstance(
                PointerAnalysis.class.getClassLoader(),
                new Class<?>[] {PointerAnalysis.class},
                (proxy, method, args) -> {
                  if (method.getName().equals("getPointerKeys")) reads.incrementAndGet();
                  try {
                    return method.invoke(pa, args);
                  } catch (InvocationTargetException e) {
                    throw e.getCause();
                  }
                });

    Map<CGNode, OrdinalSet<PointerKey>> mod = modRef.computeMod(cg, counting);

    assertEquals(Set.of("global a"), globalNames(mod.get(getNode(cg, "f"))));
    assertEquals(Set.of("global a", "global b"), globalNames(mod.get(getNode(cg, "g"))));
    assertEquals(1, reads.get());
  }

  /**
   * Several global writes find their keys from one read of the analysis's pointer keys
   * (https://github.com/wala/ML/issues/1063).
   */
  @Test
  public void testGlobalWritesReadPointerKeysOnce()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertGlobalWritesReadPointerKeysOnce(new PythonModRef());
  }

  /**
   * A subclass making its own {@link PythonModRef.PythonModVisitor}s shares the one read of the
   * analysis's pointer keys too (https://github.com/wala/ML/issues/1063).
   */
  @Test
  public void testSubclassVisitorsReadPointerKeysOnce()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    assertGlobalWritesReadPointerKeysOnce(
        new PythonModRef() {
          @Override
          protected ModVisitor<InstanceKey, ? extends ExtendedHeapModel> makeModVisitor(
              CGNode n,
              Collection<PointerKey> result,
              PointerAnalysis<InstanceKey> pa,
              ExtendedHeapModel h,
              boolean ignoreAllocHeapDefs) {
            return new PythonModVisitor<>(n, result, h, pa, ignoreAllocHeapDefs) {};
          }
        });
  }

  /**
   * A mod visitor made outside {@link ModRef#computeMod} finds a global write's keys too
   * (https://github.com/wala/ML/issues/1063).
   */
  @Test
  public void testGlobalWriteOutsideComputeMod()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("globals_two_writers.py");
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    CGNode g = getNode(cg, "g");

    Set<PointerKey> result = HashSetFactory.make();
    ExtendedHeapModel h = new PythonModRef().makeHeapModel(pa);
    SSAInstruction.Visitor visitor = new PythonModRef.PythonModVisitor<>(g, result, h, pa, false);
    for (SSAInstruction instruction : g.getIR().getInstructions())
      if (instruction != null) instruction.visit(visitor);

    assertEquals(Set.of("global a", "global b"), globalNames(result));
  }
}
