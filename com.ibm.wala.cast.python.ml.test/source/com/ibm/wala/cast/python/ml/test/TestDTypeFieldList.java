package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.util.collections.Pair;
import java.io.File;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * Witnesses for the wala/ML#865 remainder: dtype-token spellings beyond the original field list
 * resolve to their actual dtypes. One arm per (alias family, dtype) pair over the canonical
 * allocations and identity-shared aliases in {@code numpy.xml}.
 *
 * <p>The decisive arm is {@code np.int}: its truth (int64) differs from the old unconditional
 * float64 default, so int64 here proves the RESOLUTION happened rather than a default being
 * restored, where the float64-valued arms alone could pass by coincidence. The spellings whose
 * dtypes have no {@link com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType} yet ({@code
 * int16}, {@code int8}, {@code uint16}, {@code uint}, {@code float16}/{@code half}, {@code short},
 * {@code byte}) are deliberately absent pending an enum extension and stay unresolvable.
 */
public class TestDTypeFieldList extends TestPythonMLCallGraphShape {

  @Test
  public void testEachSpellingResolvesToItsActualDType() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_dtype_field_list.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    assertNotNull(builder.makeCallGraph(builder.getOptions()));
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    assertEquals(
        "Expecting np.int to resolve to int64: the truth differs from the old float64 default, so"
            + " this arm proves resolution rather than a restored fallback.",
        Set.of("int64"),
        returnedCellTypes(analysis, "via_np_int"));
    assertEquals(Set.of("float64"), returnedCellTypes(analysis, "via_np_float"));
    assertEquals(Set.of("int64"), returnedCellTypes(analysis, "via_np_long"));
    assertEquals(Set.of("string"), returnedCellTypes(analysis, "via_np_str"));
    assertEquals(Set.of("string"), returnedCellTypes(analysis, "via_np_str_"));
    assertEquals(Set.of("float32"), returnedCellTypes(analysis, "via_np_single"));
    assertEquals(Set.of("int32"), returnedCellTypes(analysis, "via_np_intc"));
    assertEquals(
        "Expecting the BUILTIN bool token to resolve through getDTypesFromDTypeArgument's builtin"
            + " arm: a report of this arm failing did not reproduce on this construct, so the arm"
            + " is pinned positively here and any failing construct needs its own reduction.",
        Set.of("bool"),
        returnedCellTypes(analysis, "via_builtin_bool"));
  }

  /**
   * The cell types carried by the named function's return value, unioned over its context-sensitive
   * nodes. Mirrors {@code TestDTypeAbsentVersusUnresolved}'s read.
   *
   * @param analysis The completed tensor type analysis.
   * @param function The function's name.
   * @return The cell type names.
   */
  private static Set<String> returnedCellTypes(TensorTypeAnalysis analysis, String function) {
    Set<String> ret = new HashSet<>();

    for (Pair<PointerKey, TensorVariable> pair : analysis) {
      String key = pair.fst.toString();

      if (!key.startsWith("[Ret-V:") || !key.contains("/" + function + ">")) continue;
      if (pair.snd == null) continue;

      for (TensorType type : pair.snd.getTypes()) ret.add(type.getCellType());
    }

    return ret;
  }
}
