package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.util.Util;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.util.intset.OrdinalSet;
import java.io.File;
import java.util.Collections;
import org.junit.Test;

/**
 * Tests the readers of a points-to set that skip the keys for points-to clients only without
 * filtering the whole set ({@link Util#onlyKeyWithoutClientOnlyKeys(OrdinalSet)}), and the parse of
 * an index that answers {@code null} rather than throwing ({@link Util#parseIntOrNull(String)}).
 * Both replace a form whose cost grew with the set's size or with the depth of the stack, so each
 * must answer exactly what the form it replaces answered.
 */
public class TestClientOnlyKeyReads extends TestPythonMLCallGraphShape {

  /** A fixture whose points-to sets hold Python scalars' type-level keys among other keys. */
  private static final String FILE = "tf2_test_shape_list_elements.py";

  /**
   * The single key {@link Util#withoutClientOnlyKeys(OrdinalSet)} keeps, as the readers asked it
   * before {@link Util#onlyKeyWithoutClientOnlyKeys(OrdinalSet)} existed.
   *
   * @param pts The points-to set.
   * @return The filtered set's only key, or {@code null} when it keeps none or several.
   */
  private static InstanceKey filteredOnlyKey(OrdinalSet<InstanceKey> pts) {
    OrdinalSet<InstanceKey> kept = Util.withoutClientOnlyKeys(pts);
    return kept == null || kept.size() != 1 ? null : kept.iterator().next();
  }

  /**
   * On every points-to set of an analyzed fixture, the single key without client-only keys is the
   * one filtering the set and counting what is left finds, including sets whose client-only keys
   * make the two counts differ.
   *
   * @throws Exception If the analysis fails.
   */
  @Test
  public void testOnlyKeyAgreesWithFiltering() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);

    int withClientOnlyKeys = 0;
    for (PointerKey pk : builder.getPointerAnalysis().getPointerKeys()) {
      OrdinalSet<InstanceKey> pts = builder.getPointerAnalysis().getPointsToSet(pk);
      assertEquals(
          "The only key of " + pk + ".",
          filteredOnlyKey(pts),
          Util.onlyKeyWithoutClientOnlyKeys(pts));
      if (Util.withoutClientOnlyKeys(pts).size() != pts.size()) withClientOnlyKeys++;
    }
    assertTrue("Expecting sets with client-only keys.", withClientOnlyKeys > 0);
    assertNull(Util.onlyKeyWithoutClientOnlyKeys(null));
    assertNull(Util.onlyKeyWithoutClientOnlyKeys(OrdinalSet.empty()));
  }

  /**
   * {@link Integer#parseInt(String)}'s answer, or {@code null} where it throws.
   *
   * @param s The string.
   * @return Its value, or {@code null}.
   */
  private static Integer parseIntCatching(String s) {
    try {
      return Integer.parseInt(s);
    } catch (NumberFormatException e) {
      return null;
    }
  }

  /** The parse answers what {@link Integer#parseInt(String)} does, and {@code null} for a throw. */
  @Test
  public void testParseIntOrNullAgreesWithParseInt() {
    String[] inputs = {
      null,
      "",
      "0",
      "7",
      "-3",
      "+12",
      "-",
      "+",
      "007",
      "1_000",
      " 1",
      "1 ",
      "x",
      "__init__",
      "shape",
      "1.5",
      "2147483647",
      "2147483648",
      "-2147483648",
      "-2147483649",
      "99999999999999999999",
      "١٢" // Arabic-Indic digits, which parseInt accepts.
    };
    for (String s : inputs)
      assertEquals("Parsing " + s + ".", parseIntCatching(s), Util.parseIntOrNull(s));
  }
}
