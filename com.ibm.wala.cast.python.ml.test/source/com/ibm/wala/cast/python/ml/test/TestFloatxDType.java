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
 * Witnesses for wala/ML#870: a {@code tf.keras.backend.floatx()} dtype token resolves to {@code
 * float32}. The call result is not a module field, so the dtype-argument resolver's field-identity
 * match cannot see it; the fix models {@code floatx()} to allocate a {@code DType} the resolver
 * recognizes by the allocating node, and the token then resolves wherever it flows.
 *
 * <p>The module-field arm is the positive control: {@code dtype=tf.float32} resolves the ordinary
 * way, and the fix must not disturb it. The formal-passing arm ({@code allocate(floatx())}) is the
 * shape that dominated the subject's degradations, an allocator helper receiving {@code floatx()}
 * as a parameter, the same one frame later as a layer {@code build} method passing {@code floatx()}
 * to {@code add_weight}. The alias arm is the spelling real subjects import with.
 */
public class TestFloatxDType extends TestPythonMLCallGraphShape {

  @Test
  public void testFloatxResolvesToFloat32() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_floatx_dtype.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    assertNotNull(builder.makeCallGraph(builder.getOptions()));
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    assertEquals(
        "Expecting the module-field control to resolve the ordinary way.",
        Set.of("float32"),
        returnedCellTypes(analysis, "via_field"));
    assertEquals(
        "Expecting a direct floatx() call result to resolve to float32, not degrade to unknown.",
        Set.of("float32"),
        returnedCellTypes(analysis, "via_floatx_call"));
    assertEquals(
        "Expecting floatx() passed as a formal to an allocating helper to resolve: this shape"
            + " dominated the subject's degradations.",
        Set.of("float32"),
        returnedCellTypes(analysis, "via_formal"));
    assertEquals(
        "Expecting floatx() reached through the module alias to resolve, the real-subject import"
            + " spelling.",
        Set.of("float32"),
        returnedCellTypes(analysis, "via_floatx_alias"));
  }

  /**
   * Witness for <a href="https://github.com/wala/ML/issues/871">wala/ML#871</a>: when {@code
   * tf.keras.backend.set_floatx} is reachable, {@code floatx()}'s value is not statically knowable,
   * so a {@code dtype=floatx()} token degrades to the unknown dtype rather than assuming the {@code
   * float32} default, which would be confidently wrong for a program that overrode it. {@link
   * #testFloatxResolvesToFloat32} is the control: without a reachable {@code set_floatx}, the
   * documented default still holds.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testFloatxDegradesWhenSetFloatxReachable() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_floatx_set_floatx.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    assertNotNull(builder.makeCallGraph(builder.getOptions()));
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    assertEquals(
        "Expecting floatx() to degrade to the unknown dtype when set_floatx is reachable, rather"
            + " than assuming the float32 default (wala/ML#871).",
        Set.of("unknown"),
        returnedCellTypes(analysis, "via_floatx_call"));
  }

  /**
   * The cell types carried by the named function's return value, unioned over its context-sensitive
   * nodes.
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
