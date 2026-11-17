package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.modref.PythonModRef;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldKey;
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
 * Tests that the constructor of {@code tf.keras.layers.Wrapper} stores the layer a subclass passes
 * it as {@code self.layer} (wala/ML#1288). Before, the {@code Wrapper} summary had no constructor,
 * so in the subclass {@code self.layer} held nothing: its call reached no target and a store into
 * one of its attributes wrote no location. {@code Held}, which stores the layer itself, is the
 * control.
 */
public class TestKerasWrapperConstructor extends AbstractTensorTest {

  private static final String FILE = "tf2_test_keras_wrapper_constructor.py";

  /** The type name of a {@code tf.keras.layers.Dense} instance. */
  private static final String DENSE = "Ltensorflow/keras/layers/Dense";

  /**
   * The result of a {@code Wrapper} subclass's call is its own wrapped layer's: each of two
   * instances, around a layer of 8 and a layer of 3 units, reads its own width.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testWrappedCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_wide", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 8))));
    test(FILE, "consume_narrow", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * A control: the result of the call of a layer that stores the wrapped layer itself.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testHeldCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_held", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5))));
  }

  /**
   * A {@code Wrapper} subclass's {@code call} writes the {@code kernel} of the layer it wraps, as a
   * mod-ref query over the call graph sees it.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testWrappedCallWritesKernel()
      throws ClassHierarchyException, CancelException, IOException {
    Set<String> modified = this.modifiedFields("Wrapped.call");
    assertTrue(modified.toString(), modified.contains(DENSE + ".kernel"));
  }

  /**
   * A control: the {@code call} of a layer that stores the wrapped layer itself writes its {@code
   * kernel}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testHeldCallWritesKernel()
      throws ClassHierarchyException, CancelException, IOException {
    Set<String> modified = this.modifiedFields("Held.call");
    assertTrue(modified.toString(), modified.contains(DENSE + ".kernel"));
  }

  /**
   * Runs the analysis on the fixture and collects the instance fields a function's nodes modify,
   * transitively through their callees, as {@code Type.field}, with the type of the instance the
   * field belongs to.
   *
   * @param function The function, as {@code Class.method}.
   * @return The modified fields.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  private Set<String> modifiedFields(String function)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    Map<CGNode, OrdinalSet<PointerKey>> mod =
        new PythonModRef().computeMod(cg, builder.getPointerAnalysis());
    String signature = "script " + FILE + "." + function + ".do()LRoot;";
    Set<String> ret = HashSetFactory.make();
    boolean reached = false;
    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().equals(signature)) continue;
      reached = true;
      for (PointerKey key : mod.get(node))
        if (key instanceof InstanceFieldKey field)
          ret.add(
              field.getInstanceKey().concreteType().getName() + "." + field.getField().getName());
    }
    assertTrue("Function must exist in call graph: " + signature, reached);
    return ret;
  }
}
