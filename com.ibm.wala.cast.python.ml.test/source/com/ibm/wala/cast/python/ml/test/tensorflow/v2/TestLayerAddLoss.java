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
 * Tests that a Keras layer whose {@code call} registers a loss with {@code self.add_loss} writes to
 * the layer, as a mod-ref query over the call graph sees it (wala/ML#1095). Registering a loss
 * appends to the layer's losses, which the training step reads back, and that write is what makes
 * the {@code call} unsafe to trace as a {@code tf.function}. Before, {@code add_loss} had no
 * summary, so the call resolved to nothing and {@code call} read as leaving the layer unchanged.
 */
public class TestLayerAddLoss extends AbstractTensorTest {

  private static final String FILE = "tf2_test_layer_add_loss.py";

  /** The layer's own {@code call} registers the loss. */
  @Test
  public void testCallAddsLoss() throws ClassHierarchyException, CancelException, IOException {
    assertTrue(this.modifiedLayerFields("Var.call").contains("Var._losses"));
  }

  /**
   * An enclosing layer's {@code call} calls the layer that registers the loss: the write to the
   * inner layer is in the enclosing {@code call}'s mod set too.
   */
  @Test
  public void testEnclosingCallAddsLoss()
      throws ClassHierarchyException, CancelException, IOException {
    assertTrue(this.modifiedLayerFields("Outer.call").contains("Var._losses"));
  }

  /**
   * A subclass of {@code tf.keras.Model} registers a loss in its own {@code call}: a model is a
   * layer at run time, so the call writes the model's {@code _losses}.
   */
  @Test
  public void testModelCallAddsLoss() throws ClassHierarchyException, CancelException, IOException {
    assertTrue(this.modifiedLayerFields("Vae.call").contains("Vae._losses"));
  }

  /** A control: a layer that registers no loss writes no losses. */
  @Test
  public void testCallWithoutLoss() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(
        Set.of(),
        this.modifiedLayerFields("Plain.call").stream()
            .filter(f -> f.endsWith("._losses"))
            .collect(java.util.stream.Collectors.toSet()));
  }

  /**
   * Runs the analysis on the fixture and collects the instance fields of layers that a function's
   * nodes modify, transitively through their callees, as {@code Class.field}.
   *
   * @param function The function, as {@code Class.method}.
   * @return The modified layer fields.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  private Set<String> modifiedLayerFields(String function)
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
        if (!(key instanceof InstanceFieldKey field)) continue;
        InstanceKey instance = field.getInstanceKey();
        if (!(instance instanceof AllocationSiteInNode allocation)) continue;
        // A program instance's concrete type is a plain object; its class is the constructor that
        // allocates it.
        String type = allocation.getNode().getMethod().getDeclaringClass().getName().toString();
        if (!type.startsWith(prefix)) continue;
        ret.add(type.substring(prefix.length()) + "." + field.getField().getName());
      }
    }
    assertTrue("Function must exist in call graph: " + signature, reached);
    return ret;
  }
}
