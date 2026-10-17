package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a dictionary copied in a loop over another dictionary's {@code items()}, as in {@code
 * for k, v in d.items(): copy[k] = v}, keeps each key paired with its own value (<a
 * href="https://github.com/wala/ML/issues/1107">wala/ML#1107</a>). Before, {@code items()} yielded
 * one tuple holding every key and every value, so the copy wrote every value under every key: a
 * {@code **} unpacking of the copy passed every value to every keyword, and a constant-key read of
 * it returned every value.
 */
public class TestItemsCopyPairing extends AbstractTensorTest {

  private static final String FILE = "tf2_test_items_copy_pairing.py";

  /**
   * A copy unpacked with {@code **} into a constructor passes each keyword parameter the value
   * stored under its own name: {@code stimuli} the {@code Emb} and {@code kernel} the {@code Gate}.
   */
  @Test
  public void testKeywordUnpacking() throws ClassHierarchyException, CancelException, IOException {
    Map<Integer, Set<String>> layers = this.parameterLayers("Net.__init__");
    // The parameters after the function object and `self`.
    assertEquals(Set.of("Emb"), layers.get(3));
    assertEquals(Set.of("Gate"), layers.get(4));
  }

  /**
   * A constant-key read of a copy returns only the value stored under that key: the path string,
   * not the tensors stored under the other keys, so the function it is passed to has no tensor.
   */
  @Test
  public void testConstantKeyRead() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "get_dataset_size", 0, 0);
  }

  /**
   * A control: a copy loop over a list of pairs, not over a dictionary's {@code items()}, still
   * takes the ordinary write of every value under every key, so the value read under {@code "x"}
   * includes the one stored under {@code "y"}.
   */
  @Test
  public void testPairsLoop() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_pair", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32, TENSOR_3_FLOAT32)));
  }

  /**
   * The program layers each parameter of a function may hold, by parameter value number, each named
   * by the class whose constructor allocates it.
   *
   * @param function The function, as {@code Class.method}.
   * @return The layer classes each parameter may hold.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the fixture cannot be read.
   */
  private Map<Integer, Set<String>> parameterLayers(String function)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    String signature = "script " + FILE + "." + function + ".do()LRoot;";
    String prefix = "Lscript " + FILE + "/";
    Map<Integer, Set<String>> ret = new java.util.HashMap<>();
    boolean reached = false;
    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().equals(signature)) continue;
      reached = true;
      for (int vn : node.getIR().getSymbolTable().getParameterValueNumbers()) {
        Set<String> layers = ret.computeIfAbsent(vn, k -> new HashSet<>());
        PointerKey key = builder.getPointerKeyForLocal(node, vn);
        for (InstanceKey instance : builder.getPointerAnalysis().getPointsToSet(key)) {
          if (!(instance instanceof AllocationSiteInNode allocation)) continue;
          // A program instance's concrete type is a plain object; its class is the constructor
          // that allocates it.
          String type = allocation.getNode().getMethod().getDeclaringClass().getName().toString();
          if (type.startsWith(prefix)) layers.add(type.substring(prefix.length()));
        }
      }
    }
    assertTrue("`" + function + "` must be in the call graph.", reached);
    return ret;
  }
}
