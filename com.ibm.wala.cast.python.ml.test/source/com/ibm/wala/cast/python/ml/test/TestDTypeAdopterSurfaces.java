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
 * Per-surface witnesses for the wala/ML#865 adopters beyond {@code NpOnes}: each surface's API
 * default survives for a determinately absent {@code dtype} argument and degrades to unknown for a
 * supplied one nothing can read. One witness pair per adopted surface, so a surface that stops
 * degrading fails its own arms rather than borrowing another's verification: {@code tf.ones} and
 * {@code tf.zeros} pin the shared {@code TensorTypeAllocator} emission, {@code np.eye} pins {@code
 * NpEye}, and {@code tf.range} pins {@code Range}'s composition, where the operand-derived result
 * (a supplied {@code dtype=} wins over operand inference at runtime) is the default only under
 * determinate absence.
 *
 * <p>The family's other non-overriding inheritors of the shared emission ({@code FixedLenFeature},
 * {@code VarLenFeature}, {@code Gamma}, {@code Poisson}, {@code Eye}, {@code SparseEye}, {@code
 * AddWeight}, {@code RandomDistribution}) inherit the degradation arm WITHOUT a witness here and
 * are named in the emission's comment so they do not borrow these surfaces' verification. {@code
 * Linspace} is deliberately absent: its modeled API has no {@code dtype} parameter, so the
 * condition cannot arise. {@code NpZeros} is covered by inheritance from the already-witnessed
 * {@code NpOnes}.
 */
public class TestDTypeAdopterSurfaces extends TestPythonMLCallGraphShape {

  @Test
  public void testEachSurfaceDefaultsOnAbsenceAndDegradesOnUnreadable() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_dtype_adopter_surfaces.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    assertNotNull(builder.makeCallGraph(builder.getOptions()));
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    assertEquals(
        "Expecting tf.ones's documented default for an absent dtype argument.",
        Set.of("float32"),
        returnedCellTypes(analysis, "ones_absent"));
    assertEquals(
        "Expecting tf.ones to degrade for a supplied dtype nothing can read, instead of the"
            + " default wearing a resolution's authority.",
        Set.of("unknown"),
        returnedCellTypes(analysis, "ones_unreadable"));
    assertEquals(
        "Expecting tf.zeros's documented default for an absent dtype argument.",
        Set.of("float32"),
        returnedCellTypes(analysis, "zeros_absent"));
    assertEquals(
        "Expecting tf.zeros to degrade for a supplied dtype nothing can read.",
        Set.of("unknown"),
        returnedCellTypes(analysis, "zeros_unreadable"));
    assertEquals(
        "Expecting np.eye's documented default for an absent dtype argument.",
        Set.of("float64"),
        returnedCellTypes(analysis, "eye_absent"));
    assertEquals(
        "Expecting np.eye to degrade for a supplied dtype nothing can read.",
        Set.of("unknown"),
        returnedCellTypes(analysis, "eye_unreadable"));
    assertEquals(
        "Expecting tf.range's operand-derived dtype for an absent dtype argument.",
        Set.of("int32"),
        returnedCellTypes(analysis, "range_absent"));
    assertEquals(
        "Expecting tf.range to degrade for a supplied dtype nothing can read: operand inference"
            + " is a valid default only when the argument is determinately absent.",
        Set.of("unknown"),
        returnedCellTypes(analysis, "range_unreadable"));
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
