package com.ibm.wala.cast.python.ir;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * Unconfounded unit tests for the pure decision helpers behind {@code @click.option} default
 * materialization (wala/ML#875, wala/ML#886). These test the translator's contiguity decision and
 * option-name rule directly, without building a call graph, so they distinguish the guard's
 * behavior rather than merely exercising it and do not depend on any function being reachable. The
 * end-to-end binding is witnessed by {@code TestClickOptionDefaults}; the non-contiguous decline is
 * covered here because its type-level effect is currently masked by a separate defect (a {@code
 * @click.argument} decorator does not resolve its decorator chain, so such a function is unreachable
 * regardless of this guard).
 */
public class ClickOptionDefaultsTranslatorTest {

  /** A contiguous trailing run of parameter indices is kept whole. */
  @Test
  public void contiguousTrailingBlockKeepsAContiguousTrailingRun() {
    Map<Integer, String> all = Map.of(1, "a", 2, "b", 3, "c");
    assertEquals(all, PythonCAstToIRTranslator.contiguousTrailingBlock(all, 4, 0));

    // A single trailing option.
    assertEquals(
        Map.of(3, "c"), PythonCAstToIRTranslator.contiguousTrailingBlock(Map.of(3, "c"), 4, 0));

    // A click option sitting immediately below a Python default (index 2 is the Python default,
    // index 1 the click option) is still contiguous with it.
    assertEquals(
        Map.of(1, "a"), PythonCAstToIRTranslator.contiguousTrailingBlock(Map.of(1, "a"), 3, 1));
  }

  /** A non-contiguous set declines whole, rather than partially materializing. */
  @Test
  public void contiguousTrailingBlockDeclinesANonContiguousSet() {
    // A gap at index 2 (e.g. a @click.argument between two options): not a contiguous block.
    assertTrue(
        PythonCAstToIRTranslator.contiguousTrailingBlock(Map.of(1, "a", 3, "c"), 4, 0).isEmpty());

    // An option (index 1) sitting ABOVE a required argument (index 2, no default): declines, so its
    // default is never bound to the wrong parameter. This is the case the fixture's `declined` has.
    assertTrue(PythonCAstToIRTranslator.contiguousTrailingBlock(Map.of(1, "a"), 3, 0).isEmpty());

    // Empty in, empty out.
    assertTrue(
        PythonCAstToIRTranslator.contiguousTrailingBlock(Map.<Integer, String>of(), 4, 0)
            .isEmpty());
  }

  /** The option-to-parameter name rule: long spelling wins, dashes become underscores. */
  @Test
  public void clickOptionParameterNameFollowsClicksRule() {
    assertEquals(
        "vocab_size", PythonCAstToIRTranslator.clickOptionParameterName(List.of("--vocab-size")));

    // The long spelling wins over a short one, whatever the order.
    assertEquals(
        "num_layers",
        PythonCAstToIRTranslator.clickOptionParameterName(List.of("-n", "--num-layers")));

    // A short-only option falls back to the short letter.
    assertEquals("x", PythonCAstToIRTranslator.clickOptionParameterName(List.of("-x")));

    // Nothing option-shaped yields null.
    assertNull(PythonCAstToIRTranslator.clickOptionParameterName(List.of("notaflag")));
    assertNull(PythonCAstToIRTranslator.clickOptionParameterName(List.of()));
  }
}
