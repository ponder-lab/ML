package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.TensorGenerator.TypeFeed;
import com.ibm.wala.cast.python.ml.client.TensorGenerator.TypeFeedKind;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.cast.python.ml.util.TensorShapeUtil.MatMulReading;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * A matmul's type feed reads the operand flags as its generator does (<a
 * href="https://github.com/wala/ML/issues/1164">wala/ML#1164</a>). The feed composes the product
 * when the generator cannot read an operand, so a feed composing its operands unswapped gave a
 * flagged call a wrong shape member there.
 */
public class MatMulTypeFeedTest extends TestPythonMLCallGraphShape {

  /**
   * The feed of a {@code transpose_b=True} matmul carries the one reading that transposes {@code
   * b}, the reading its generator composes over.
   */
  @Test
  public void testFeedCarriesTheFlagReadings() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_matmul_feed_flags.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    try {
      List<TypeFeed> feeds = new ArrayList<>();
      for (CGNode node : builder.getCallGraph()) {
        if (node.getIR() == null
            || !node.getMethod().getDeclaringClass().getName().toString().startsWith("Lscript "))
          continue;
        for (SSAInstruction instruction : node.getIR().getInstructions()) {
          if (!(instruction instanceof SSAAbstractInvokeInstruction call) || !call.hasDef())
            continue;
          PointerKey result =
              builder
                  .getPointerAnalysis()
                  .getHeapModel()
                  .getPointerKeyForLocal(node, call.getDef());
          TensorGenerator generator;
          try {
            generator =
                TensorGeneratorFactory.getGenerator(
                    builder.getPropagationSystem().findOrCreatePointsToSet(result), builder);
          } catch (IllegalArgumentException e) {
            continue;
          }
          if (generator instanceof MatMul) feeds.add(generator.getTypeFeed(builder));
        }
      }
      // The matmul's result and the result of the sink that returns it both dispatch to its
      // generator.
      assertFalse("Expecting the script's matmul.", feeds.isEmpty());
      for (TypeFeed feed : feeds) {
        assertEquals(TypeFeedKind.MATMUL, feed.kind());
        assertEquals(
            "The feed reads the flags as the generator does.",
            Set.of(new MatMulReading(false, true)),
            feed.matmulReadings());
      }
    } finally {
      TensorGenerator.clearCaches(builder);
    }
  }
}
