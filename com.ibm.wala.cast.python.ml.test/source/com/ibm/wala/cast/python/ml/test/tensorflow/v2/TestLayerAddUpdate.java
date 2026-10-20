package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.modref.PythonModRef;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that applying a {@code BatchNormalization} layer, or calling {@code Layer.add_update},
 * writes the layer's {@code _updates}, as a mod-ref query over the call graph sees it, and that the
 * normalization layer receives the {@code training} value it is applied with (wala/ML#1119). The
 * layer registers the update of its moving statistics with {@code add_update}, and Keras skips such
 * an update while it builds a graph, so a client deciding whether a function is safe to trace needs
 * to see it. Before, {@code add_update} had no summary and the normalization summaries wrote
 * nothing and took no {@code training}.
 */
public class TestLayerAddUpdate extends AbstractTensorTest {

  private static final String FILE = "tf2_test_layer_add_update.py";

  /** The type of the normalization layer's instances. */
  private static final String BATCH_NORMALIZATION = "Ltensorflow/keras/layers/BatchNormalization";

  /** The type of the simple recurrent layer's instances. */
  private static final String SIMPLE_RNN = "Ltensorflow/keras/layers/SimpleRNN";

  /** A layer's {@code call} that applies a normalization layer writes that layer's updates. */
  @Test
  public void testCallAppliesNormalization()
      throws ClassHierarchyException, CancelException, IOException {
    assertTrue(
        this.updatedLayers("Block.call").toString(),
        this.updatedLayers("Block.call").contains(BATCH_NORMALIZATION));
  }

  /** A layer's {@code call} that registers an update with {@code add_update} writes itself. */
  @Test
  public void testCallAddsUpdate() throws ClassHierarchyException, CancelException, IOException {
    assertTrue(
        this.updatedLayers("Counter.call").toString(),
        this.updatedLayers("Counter.call").contains("Counter"));
  }

  /** A plain function that applies a normalization layer writes that layer's updates. */
  @Test
  public void testFunctionAppliesNormalization()
      throws ClassHierarchyException, CancelException, IOException {
    assertTrue(
        this.updatedLayers("normalize_training").toString(),
        this.updatedLayers("normalize_training").contains(BATCH_NORMALIZATION));
  }

  /**
   * A control: a recurrent layer that is not stateful registers no update, so applying it writes no
   * {@code _updates}.
   */
  @Test
  public void testRecurrentRegistersNoUpdate()
      throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(), this.updatedLayers("Recurrent.call"));
  }

  /**
   * A stateful recurrent layer registers the update of its state with {@code add_update}, so
   * applying it writes the layer's {@code _updates}.
   *
   * <p>TODO: The recurrent layers' summaries write nothing, so this pins a known gap. Remove the
   * expected {@link AssertionError} once <a
   * href="https://github.com/wala/ML/issues/1124">wala/ML#1124</a> is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testStatefulRecurrentRegistersUpdate()
      throws ClassHierarchyException, CancelException, IOException {
    assertTrue(
        this.updatedLayers("StatefulRecurrent.call").toString(),
        this.updatedLayers("StatefulRecurrent.call").contains(SIMPLE_RNN));
  }

  /**
   * A normalization layer applied with {@code training=False} registers no update, so applying it
   * in inference writes no {@code _updates}.
   *
   * <p>TODO: The summary cannot branch on {@code training} and writes on every application, so this
   * pins a known over-approximation. Remove the expected {@link AssertionError} once <a
   * href="https://github.com/wala/ML/issues/1124">wala/ML#1124</a> is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testInferenceNormalizationRegistersNoUpdate()
      throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(), this.updatedLayers("normalize_inference"));
  }

  /** A control: a layer that applies no stateful layer registers no update. */
  @Test
  public void testPlainRegistersNoUpdate()
      throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(), this.updatedLayers("Plain.call"));
  }

  /**
   * The normalization layer's {@code __call__} takes {@code training}, so it receives the values it
   * is applied with: {@code True} from {@code normalize_training} and {@code False} from {@code
   * normalize_inference}. The summary's node serves every application, through a trampoline per
   * call site, so its parameter holds both; the value one application passes is the caller's
   * argument.
   */
  @Test
  public void testNormalizationReceivesTraining()
      throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(true, false), this.normalizationTraining());
  }

  /**
   * Runs the analysis on the fixture and collects the layers whose {@code _updates} a function's
   * nodes modify, transitively through their callees: a program layer by its class's name in the
   * fixture, a library layer by its type.
   *
   * @param function The function, as {@code Class.method} or a function's name.
   * @return The updated layers.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  private Set<String> updatedLayers(String function)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    Map<CGNode, OrdinalSet<PointerKey>> mod =
        new PythonModRef().computeMod(cg, builder.getPointerAnalysis());

    String signature = "script " + FILE + "." + function + ".do()LRoot;";
    String prefix = "Lscript " + FILE + "/";
    Set<String> ret = HashSetFactory.make();
    boolean reached = false;
    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().equals(signature)) continue;
      reached = true;
      for (PointerKey key : mod.get(node)) {
        if (!(key instanceof InstanceFieldKey field)
            || !field.getField().getName().toString().equals("_updates")) continue;
        InstanceKey instance = field.getInstanceKey();
        String type = instance.concreteType().getName().toString();
        // A program instance's concrete type is a plain object; its class is the constructor that
        // allocates it.
        if (instance instanceof AllocationSiteInNode allocation) {
          String constructor =
              allocation.getNode().getMethod().getDeclaringClass().getName().toString();
          if (constructor.startsWith(prefix)) type = constructor.substring(prefix.length());
        }
        ret.add(type);
      }
    }
    assertTrue("Function must exist in call graph: " + signature, reached);
    return ret;
  }

  /**
   * Runs the analysis on the fixture and collects the constants the {@code training} parameter of
   * the normalization layer's {@code __call__} summary receives.
   *
   * @return The constants.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  private Set<Object> normalizationTraining()
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    Set<Object> ret = HashSetFactory.make();
    boolean reached = false;
    for (CGNode node : cg) {
      if (!node.getMethod()
          .getDeclaringClass()
          .getName()
          .toString()
          .equals(BATCH_NORMALIZATION + "/__call__")) continue;
      reached = true;
      // The parameters are the function object, `self`, `inputs` and `training`.
      assertTrue("The summary takes training.", node.getIR().getNumberOfParameters() > 3);
      PointerKey training =
          builder
              .getPointerAnalysis()
              .getHeapModel()
              .getPointerKeyForLocal(node, node.getIR().getParameter(3));
      for (InstanceKey instance : builder.getPointerAnalysis().getPointsToSet(training))
        if (instance instanceof ConstantKey<?> constant) ret.add(constant.getValue());
    }
    assertTrue("The normalization layer's call must exist in the call graph.", reached);
    return ret;
  }
}
