package com.ibm.wala.cast.python.ml.test;

import static java.util.Arrays.asList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.util.collections.Pair;
import java.io.File;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * Witnesses for wala/ML#874: {@code tf.ones_like}'s result has the input's shape, its dtype
 * defaults to the input's, and an explicit {@code dtype=} overrides that default.
 *
 * <p>The override arm is load-bearing: the prior {@code pass_through} model returned the input, so
 * {@code tf.ones_like(x, dtype=tf.int32)} on a float32 {@code x} read {@code float32}, confidently
 * wrong on a value whose program text names the dtype. Modeling it as an allocation with the {@code
 * OnesLike} generator, as {@code tf.zeros_like} already is, lets the explicit argument win.
 */
public class TestOnesLike extends TestPythonMLCallGraphShape {

  @Test
  public void testOnesLikeShapeAndDtype() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_ones_like.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    assertNotNull(builder.makeCallGraph(builder.getOptions()));
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    assertEquals(
        "Expecting an omitted dtype to inherit the input's float32, over the input's shape.",
        Set.of(new TensorType("float32", asList(new NumericDim(3), new NumericDim(4)))),
        returnedTypes(analysis, "ones_like_default"));
    assertEquals(
        "Expecting an explicit dtype=tf.int32 to override the input's float32: the pass-through"
            + " model read float32 here, confidently wrong.",
        Set.of(new TensorType("int32", asList(new NumericDim(3), new NumericDim(4)))),
        returnedTypes(analysis, "ones_like_override"));
  }

  /**
   * The tensor types carried by the named function's return value, unioned over its
   * context-sensitive nodes.
   *
   * @param analysis The completed tensor type analysis.
   * @param function The function's name.
   * @return The types.
   */
  private static Set<TensorType> returnedTypes(TensorTypeAnalysis analysis, String function) {
    Set<TensorType> ret = new HashSet<>();

    for (Pair<PointerKey, TensorVariable> pair : analysis) {
      String key = pair.fst.toString();

      if (!key.startsWith("[Ret-V:") || !key.contains("/" + function + ">")) continue;
      if (pair.snd == null) continue;

      ret.addAll(pair.snd.getTypes());
    }

    return ret;
  }
}
