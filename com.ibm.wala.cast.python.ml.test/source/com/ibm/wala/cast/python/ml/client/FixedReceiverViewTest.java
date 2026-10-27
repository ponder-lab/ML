package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;

/**
 * Views of a transformation fixed to one receiver each ({@link DatasetGenerator#overReceiver}):
 * they compare equal only over the same receiver (wala/ML#1088), and their memoization keys tell
 * them apart, and apart from the unfixed original (wala/ML#1089).
 */
public class FixedReceiverViewTest extends TestPythonMLCallGraphShape {

  /**
   * The {@code shuffle} transformation over either of two datasets, read at {@code consume}'s
   * parameter.
   *
   * @param builder The builder.
   * @return The transformation's generator.
   */
  private static DatasetGenerator shuffled(PythonSSAPropagationCallGraphBuilder builder) {
    List<PointerKey> parameters = new ArrayList<>();
    for (CGNode node : builder.getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/consume"))
        parameters.add(builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2));
    assertEquals("Expecting one node for consume.", 1, parameters.size());
    TensorGenerator generator =
        TensorGeneratorFactory.getGenerator(
            builder.getPropagationSystem().findOrCreatePointsToSet(parameters.get(0)), builder);
    assertTrue(
        "The parameter is the transformation's dataset: " + generator,
        generator instanceof DatasetGenerator);
    return (DatasetGenerator) generator;
  }

  /**
   * Views over different receivers are different, and keyed apart from each other and the original.
   */
  @Test
  public void testViewsOverDifferentReceivers() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_dataset_fixed_receiver_views.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    try {
      DatasetGenerator original = shuffled(builder);
      List<TensorGenerator> receivers = original.getReceiverGenerators(builder);
      assertNotNull("Both receivers resolve.", receivers);
      assertEquals("The receiver is either dataset.", 2, receivers.size());

      DatasetGenerator wide = original.overReceiver(receivers.get(0));
      DatasetGenerator narrow = original.overReceiver(receivers.get(1));
      DatasetGenerator wideAgain = original.overReceiver(receivers.get(0));

      assertNotEquals("Views over different receivers differ.", wide, narrow);
      assertNotEquals("A view differs from its unfixed original.", wide, original);
      assertEquals("Views over the same receiver are equal.", wide, wideAgain);
      assertEquals("Equal views hash alike.", wide.hashCode(), wideAgain.hashCode());

      assertNotEquals(
          "Views over different receivers are keyed apart.",
          TensorGenerator.evaluationKey(wide),
          TensorGenerator.evaluationKey(narrow));
      assertNotEquals(
          "A view is keyed apart from its unfixed original.",
          TensorGenerator.evaluationKey(wide),
          TensorGenerator.evaluationKey(original));
      assertEquals(
          "Views over the same receiver share a key.",
          TensorGenerator.evaluationKey(wide),
          TensorGenerator.evaluationKey(wideAgain));
      assertEquals(
          "Keys that are equal hash alike.",
          TensorGenerator.evaluationKey(wide).hashCode(),
          TensorGenerator.evaluationKey(wideAgain).hashCode());
    } finally {
      TensorGenerator.clearCaches(builder);
    }
  }
}
