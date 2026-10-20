package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
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
import java.lang.ref.WeakReference;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
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

  /**
   * A pointer analysis that delegates to the given one, counting its reads of the pointer keys and
   * leaving out the keys the given filter rejects.
   *
   * @param pa The pointer analysis to delegate to.
   * @param reads The count of {@link PointerAnalysis#getPointerKeys()} calls.
   * @param keep The pointer keys {@link PointerAnalysis#getPointerKeys()} returns.
   * @return The delegating analysis.
   */
  private static PointerAnalysis<InstanceKey> delegating(
      PointerAnalysis<InstanceKey> pa, AtomicInteger reads, Predicate<PointerKey> keep) {
    @SuppressWarnings("unchecked")
    PointerAnalysis<InstanceKey> ret =
        (PointerAnalysis<InstanceKey>)
            Proxy.newProxyInstance(
                PointerAnalysis.class.getClassLoader(),
                new Class<?>[] {PointerAnalysis.class},
                (proxy, method, args) -> {
                  if (method.getName().equals("getPointerKeys")) {
                    reads.incrementAndGet();
                    Set<PointerKey> keys = HashSetFactory.make();
                    for (PointerKey key : pa.getPointerKeys()) if (keep.test(key)) keys.add(key);
                    return keys;
                  }
                  try {
                    return method.invoke(pa, args);
                  } catch (InvocationTargetException e) {
                    throw e.getCause();
                  }
                });
    return ret;
  }

  /**
   * The union of the mod sets {@link ModRef#getMod(CGNode, ExtendedHeapModel, PointerAnalysis,
   * SSAInstruction, com.ibm.wala.ipa.slicer.HeapExclusions)} gives each of the node's instructions,
   * the path a client computing a node's direct mod set one instruction at a time takes.
   *
   * @param modRef The mod-ref analysis.
   * @param node The node.
   * @param pa The pointer analysis.
   * @return The keys the node's instructions modify.
   */
  private static Set<PointerKey> perInstructionMod(
      ModRef<InstanceKey> modRef, CGNode node, PointerAnalysis<InstanceKey> pa) {
    ExtendedHeapModel h = modRef.makeHeapModel(pa);
    Set<PointerKey> ret = HashSetFactory.make();
    for (SSAInstruction instruction : node.getIR().getInstructions())
      if (instruction != null) ret.addAll(modRef.getMod(node, h, pa, instruction, null));
    return ret;
  }

  /**
   * Global writes visited one instruction at a time, by visitors of several {@link PythonModRef}s,
   * share one read of the analysis's pointer keys, and each function's mod set is the one {@link
   * ModRef#computeMod} gives (https://github.com/wala/ML/issues/1063).
   */
  @Test
  public void testPerInstructionGlobalWritesReadPointerKeysOnce()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("globals_two_writers.py");
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    AtomicInteger reads = new AtomicInteger();
    PointerAnalysis<InstanceKey> pa = delegating(builder.getPointerAnalysis(), reads, key -> true);
    CGNode f = getNode(cg, "f");
    CGNode g = getNode(cg, "g");

    for (CGNode node : cg) perInstructionMod(new PythonModRef(), node, pa);
    Set<PointerKey> fMod = perInstructionMod(new PythonModRef(), f, pa);
    Set<PointerKey> gMod = perInstructionMod(new PythonModRef(), g, pa);

    assertEquals(1, reads.get());
    assertEquals(Set.of("global a"), globalNames(fMod));
    assertEquals(Set.of("global a", "global b"), globalNames(gMod));

    Map<CGNode, OrdinalSet<PointerKey>> mod = new PythonModRef().computeMod(cg, pa);
    for (CGNode node : List.of(f, g)) {
      Set<PointerKey> computed = HashSetFactory.make();
      mod.get(node).forEach(computed::add);
      assertEquals(computed, node == f ? fMod : gMod);
    }
  }

  /**
   * A global with no static field key in the analysis modifies no key
   * (https://github.com/wala/ML/issues/1063).
   */
  @Test
  public void testGlobalWithoutStaticFieldKeyModifiesNothing()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("globals_two_writers.py");
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    PointerAnalysis<InstanceKey> pa =
        delegating(
            builder.getPointerAnalysis(),
            new AtomicInteger(),
            key ->
                !(key instanceof StaticFieldKey
                    && ((StaticFieldKey) key).getField().getName().toString().equals("global b")));

    assertEquals(
        Set.of("global a"),
        globalNames(perInstructionMod(new PythonModRef(), getNode(cg, "g"), pa)));
  }

  /**
   * Indexes the static fields of a fresh analysis delegating to the given one, and returns a weak
   * reference to that analysis, so that nothing else holds it.
   *
   * @param pa The pointer analysis to delegate to.
   * @param node A node whose global writes are visited.
   * @return A weak reference to the indexed analysis.
   */
  private static WeakReference<PointerAnalysis<InstanceKey>> indexFreshAnalysis(
      PointerAnalysis<InstanceKey> pa, CGNode node) {
    PointerAnalysis<InstanceKey> fresh = delegating(pa, new AtomicInteger(), key -> true);
    assertEquals(
        Set.of("global a", "global b"),
        globalNames(perInstructionMod(new PythonModRef(), node, fresh)));
    return new WeakReference<>(fresh);
  }

  /**
   * The static-field index does not keep its analysis reachable
   * (https://github.com/wala/ML/issues/1063).
   */
  @Test
  public void testStaticFieldIndexDoesNotPinItsAnalysis()
      throws ClassHierarchyException,
          IllegalArgumentException,
          CancelException,
          IOException,
          InterruptedException {
    PythonAnalysisEngine<?> engine = makeEngine("globals_two_writers.py");
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    WeakReference<PointerAnalysis<InstanceKey>> ref =
        indexFreshAnalysis(builder.getPointerAnalysis(), getNode(cg, "g"));

    for (int i = 0; i < 50 && ref.get() != null; i++) {
      System.gc();
      Thread.sleep(20);
    }

    assertNull("The index keeps an unreachable analysis.", ref.get());
  }
}
