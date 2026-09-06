package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Witnesses for <a href="https://github.com/wala/ML/issues/875">wala/ML#875</a> and <a
 * href="https://github.com/wala/ML/issues/886">wala/ML#886</a>: a {@code @click.option}'s default is
 * materialized as its parameter's default, so an unpassed option resolves inside the body to a
 * concrete tensor dimension. This is the gpt-2 {@code train} shape, whose parameters are all {@code
 * @click.option} and are read as dimensions of tensors the model builds. The materialization reuses
 * the <a href="https://github.com/wala/ML/issues/743">wala/ML#743</a> default-parameter globals
 * written by {@code PythonCAstToIRTranslator}.
 *
 * <p>The fixture's option defaults are DISTINCT and NOT ascending (30, 10, 20) precisely so a
 * reversed or positional option-to-parameter mapping fails these assertions rather than passing by
 * coincidence: click applies decorators bottom-up, so the source order of the option lines is the
 * reverse of the parameter order, and only a by-name match lands each default on its own parameter.
 *
 * <p>The non-contiguous decline (an option sitting above a {@code @click.argument}, whose default
 * must not be materialized lest a positional read misbind it) has no type-level arm here. Its
 * type-level effect would be observable in principle (without the guard the misaligned defaults
 * materialize, arity is satisfied, and the function becomes reachable carrying wrong constants), but
 * it is currently masked by a separate defect, <a
 * href="https://github.com/wala/ML/issues/891">wala/ML#891</a>: a {@code @click.argument} decorator
 * does not resolve its decorator chain, so such a function is unreachable regardless of this guard,
 * which would make a call-graph-absence assertion pass for the wrong reason. The decline is instead
 * unit-tested unconfounded in {@code ClickOptionDefaultsTranslatorTest}, and its branch is covered
 * by the fixture's {@code declined} function and its FINE "Declining to materialize" log.
 */
public class TestClickOptionDefaults extends AbstractTensorTest {

  private static final String FILE = "tf2_test_click_option_bind.py";

  /**
   * Each option's default binds to the parameter click's rule names, not the one its source
   * position would suggest. Distinct, non-ascending defaults make a mis-mapping observable.
   */
  @Test
  public void testOptionDefaultBindsByName()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_alpha", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 30))));
    test(FILE, "consume_beta", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 10))));
    test(FILE, "consume_gamma", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 20))));
  }

  /**
   * The materialized default flows through a stored-attribute chain (a {@code Projector} holding
   * the size and reshaping by it), as it does in the subject, resolving the reshape to a concrete
   * shape.
   */
  @Test
  public void testOptionDefaultFlowsThroughChain()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_chain", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 30))));
  }

  /**
   * A value supplied at the call wins over the option default: the materialized default (768) must
   * not override the passed argument (999). This is the wala/ML#743 "supplied wins" semantics,
   * which is why materializing the default is sound where injecting it into the parameter's
   * points-to set would not be.
   */
  @Test
  public void testSuppliedArgumentWinsOverOptionDefault()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_supplied", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 999))));
  }
}
