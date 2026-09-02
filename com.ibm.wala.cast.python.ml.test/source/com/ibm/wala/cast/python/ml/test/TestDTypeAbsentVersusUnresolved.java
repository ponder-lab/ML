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
 * An absent dtype argument and a supplied-but-unresolved one are different states (<a
 * href="https://github.com/wala/ML/issues/865">wala/ML#865</a>).
 *
 * <p>They want opposite answers. An argument nobody supplies takes the API default, and for {@code
 * np.ones} that default is {@code float64} as a matter of documented fact. An argument that IS
 * supplied but whose value the analysis could not read has no evidence at all, and answering it
 * with the same default asserts a guess with the authority of a resolution: nothing downstream can
 * then separate it from a dtype that genuinely resolved to {@code float64}.
 *
 * <p>A points-to set cannot tell the two apart, because it is empty in both cases. Presence is a
 * fact about the program text, so the fix reads it from the call sites rather than inferring it
 * from an absence of members.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestDTypeAbsentVersusUnresolved extends TestPythonMLCallGraphShape {

  /**
   * The four states a dtype argument can be in, each asserted separately.
   *
   * <p>Three of the four must NOT move, and they are the reason this is a four-way assertion rather
   * than one. Fixing the unresolved case by degrading every empty points-to set to ⊤ would break
   * {@code absent}, where the default is correct, and {@code supplied_none}, where an explicit
   * {@code None} genuinely does mean "use the default". A test that pinned only the defect would
   * pass for a change that traded it for those two regressions.
   *
   * <p>Verified as a pair rather than only in the passing direction: with the fix reverted, {@code
   * supplied_but_unresolved} reads {@code float64}, which is exactly the reported defect and is
   * indistinguishable from the two cases where {@code float64} is right.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testAbsentKeepsTheDefaultAndUnresolvedDegrades() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_dtype_absent_vs_unresolved.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    assertNotNull(builder.makeCallGraph(builder.getOptions()));
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    assertEquals(
        "No dtype argument, so NumPy's documented default applies and is correct.",
        Set.of("float64"),
        returnedCellTypes(analysis, "absent"));
    assertEquals(
        "A readable dtype argument decides.",
        Set.of("int32"),
        returnedCellTypes(analysis, "supplied_and_resolvable"));
    assertEquals(
        "Supplied but unreadable: there is no evidence, so the dtype is unknown rather than the"
            + " API default, which would be a guess wearing a resolution's authority.",
        Set.of("unknown"),
        returnedCellTypes(analysis, "supplied_but_unresolved"));
    assertEquals(
        "An explicit `None` really does mean \"use the default\", so this one must not degrade.",
        Set.of("float64"),
        returnedCellTypes(analysis, "supplied_none"));
    assertEquals(
        "A starred unpack makes presence INDETERMINATE, and indeterminate declines the default:"
            + " asserting it on a call shape that cannot show whether the argument is there would"
            + " re-open the defect for exactly the shapes least likely to have witnesses.",
        Set.of("unknown"),
        returnedCellTypes(analysis, "supplied_through_star"));
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

      // The return-value key of the function's own node, which is what a consumer reads.
      if (!key.startsWith("[Ret-V:") || !key.contains("/" + function + ">")) continue;
      if (pair.snd == null) continue;

      for (TensorType type : pair.snd.getTypes()) ret.add(type.getCellType());
    }

    return ret;
  }
}
