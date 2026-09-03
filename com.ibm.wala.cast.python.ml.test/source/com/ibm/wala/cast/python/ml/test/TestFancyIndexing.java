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
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.util.collections.Pair;
import java.io.File;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * Witnesses for wala/ML#866: a subscript whose member is itself an array is FANCY indexing, whose
 * result replaces the receiver's leading axis with the index's shape, while a constant-index
 * subscript keeps element semantics (peel the leading axis).
 *
 * <p>The contrast is the load-bearing pair: {@code np.eye(3)[np.array([0, 1, 2])]} and {@code
 * np.eye(3)[0]} produced the same {@code (3,)} before the fix, and only the first was wrong, so a
 * fix that merely stopped peeling would correct one spelling and break the other. The rank-3 and
 * non-square arms pin that the composition ({@code indexShape ++ receiverShape[1:]}) is general
 * rather than eye-shaped, and the boolean-mask arm pins the honest form for a selected count the
 * analysis cannot compute: the receiver's rank with an {@link UnresolvedDim} leading axis
 * (wala/ML#721's criterion, a fixed runtime integer the analysis could not compute).
 */
public class TestFancyIndexing extends TestPythonMLCallGraphShape {

  private static final String F64 = "float64";

  @Test
  public void testFancyIndexingReplacesTheLeadingAxis() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_fancy_index.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    assertNotNull(builder.makeCallGraph(builder.getOptions()));
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    assertEquals(
        "Expecting the array index's shape to replace the receiver's leading axis: the wala/ML#866"
            + " defect emitted (3,) here, indistinguishable from a genuine rank-1 resolution.",
        Set.of(new TensorType(F64, asList(new NumericDim(3), new NumericDim(3)))),
        returnedTypes(analysis, "eye_fancy"));
    assertEquals(
        "Expecting the scalar index to keep element semantics: it produces the same output the"
            + " defect did for the array spelling, and here it is correct, so this arm fails any"
            + " fix that merely stops peeling.",
        Set.of(new TensorType(F64, asList(new NumericDim(3)))),
        returnedTypes(analysis, "eye_scalar"));
    assertEquals(
        "Expecting the bare rank-3 receiver to type on its own: without this control the rank-3"
            + " arm below cannot discriminate anything.",
        Set.of(
            new TensorType(F64, asList(new NumericDim(2), new NumericDim(3), new NumericDim(4)))),
        returnedTypes(analysis, "ones3_bare"));
    assertEquals(
        "Expecting the rank-3 fancy index to compose indexShape ++ receiverShape[1:]: element"
            + " semantics would give (3, 4) and fancy indexing gives (2, 3, 4).",
        Set.of(
            new TensorType(F64, asList(new NumericDim(2), new NumericDim(3), new NumericDim(4)))),
        returnedTypes(analysis, "ones3_fancy"));
    assertEquals(
        "Expecting the non-eye rank-2 fancy index to compose as well: the defect was never"
            + " eye-specific.",
        Set.of(new TensorType(F64, asList(new NumericDim(2), new NumericDim(3)))),
        returnedTypes(analysis, "ones2_fancy"));
    assertEquals(
        "Expecting a rank-1 boolean mask to keep the receiver's rank with the selected count"
            + " unresolved: it is a fixed runtime integer the analysis cannot compute.",
        Set.of(new TensorType(F64, asList(UnresolvedDim.INSTANCE, new NumericDim(3)))),
        returnedTypes(analysis, "mask_fancy"));
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
