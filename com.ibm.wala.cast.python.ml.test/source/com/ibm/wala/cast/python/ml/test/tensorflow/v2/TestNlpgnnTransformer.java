package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.test.categories.WholeProjectFixtures;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.experimental.categories.Category;

/**
 * In-vivo exact-set pin for the NLPGNN BERT encoder loop ({@code Transformer.call} in {@code
 * nlpgnn/layers/transformer.py}), the wala/ML#753 witness. Deliberately a single-test class: the
 * pinned unions are deterministic in a fresh JVM but can flip under same-JVM predecessor analyses,
 * whose {@code identityHashCode}/ASLR variance perturbs WALA-internal collection orders
 * (wala/ML#753) &mdash; so this module's Surefire configuration forks a fresh JVM per test class,
 * and the class gets its own wala/ML#755 CI shard. Do not add further tests to this class: a
 * same-JVM predecessor analysis would reintroduce the flake.
 */
@Category(WholeProjectFixtures.class)
public class TestNlpgnnTransformer extends AbstractTensorTest {

  /**
   * Pins the {@code Transformer.call} parameter unions the encoder loop settles on. PR
   * ponder-lab/ML#629 (wala/ML#706) had to drop this pin as order-flaky; the wala/ML#753 arc (PRs
   * ponder-lab/ML#650&ndash;659) made the analysis deterministic per configuration, leaving only
   * the JVM-history carrier the per-class fork removes.
   *
   * <p>The {@code input_tensor} (value number 3) union carries one {@code (batch, seq, U)} member
   * per entry pipeline reaching the encoder &mdash; the {@code (2, 4)}, {@code (2, 10)}, {@code (8,
   * 10)}, {@code (8, 100)}, {@code (6, 128)}, and {@code (16, 100)} leading pairs from the entry
   * scripts' {@code model.build} contracts (wala/ML#717) through the embedding's output reshape,
   * each trailing hidden dimension a config-derived {@link UnresolvedDim} (wala/ML#721) &mdash;
   * plus each pair's loop-carried degraded-rank siblings ({@code (batch, U)} and {@code (batch, U,
   * U)} from the {@code reshape_to_matrix}/{@code reshape_from_matrix} round trip's non-entry
   * contexts) and the fully-unresolved rank-2/rank-3 members. The {@code mask} (value number 4)
   * union is the attention mask's {@code (batch, seq, seq)} broadcast per the same six entry
   * pipelines.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullTransformerInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // LinkedHashMap fixes the assertion order (value number 3 before 4) so a regression names the
    // same value number on every run; Map.of iteration order varies per JVM.
    Map<Integer, Set<TensorType>> expected = new LinkedHashMap<>();

    expected.put(
        3,
        Set.of(
            new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
            new TensorType(FLOAT_32, asList(new NumericDim(2), UnresolvedDim.INSTANCE)),
            new TensorType(FLOAT_32, asList(new NumericDim(6), UnresolvedDim.INSTANCE)),
            new TensorType(FLOAT_32, asList(new NumericDim(8), UnresolvedDim.INSTANCE)),
            new TensorType(FLOAT_32, asList(new NumericDim(16), UnresolvedDim.INSTANCE)),
            new TensorType(
                FLOAT_32,
                asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
            new TensorType(
                FLOAT_32,
                asList(new NumericDim(2), UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
            new TensorType(
                FLOAT_32, asList(new NumericDim(2), new NumericDim(4), UnresolvedDim.INSTANCE)),
            new TensorType(
                FLOAT_32, asList(new NumericDim(2), new NumericDim(10), UnresolvedDim.INSTANCE)),
            new TensorType(
                FLOAT_32,
                asList(new NumericDim(6), UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
            new TensorType(
                FLOAT_32, asList(new NumericDim(6), new NumericDim(128), UnresolvedDim.INSTANCE)),
            new TensorType(
                FLOAT_32,
                asList(new NumericDim(8), UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
            new TensorType(
                FLOAT_32, asList(new NumericDim(8), new NumericDim(10), UnresolvedDim.INSTANCE)),
            new TensorType(
                FLOAT_32, asList(new NumericDim(8), new NumericDim(100), UnresolvedDim.INSTANCE)),
            new TensorType(
                FLOAT_32,
                asList(new NumericDim(16), UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
            new TensorType(
                FLOAT_32,
                asList(new NumericDim(16), new NumericDim(100), UnresolvedDim.INSTANCE))));

    expected.put(
        4,
        Set.of(
            new TensorType(
                FLOAT_32, asList(new NumericDim(2), new NumericDim(4), new NumericDim(4))),
            new TensorType(
                FLOAT_32, asList(new NumericDim(2), new NumericDim(10), new NumericDim(10))),
            new TensorType(
                FLOAT_32, asList(new NumericDim(6), new NumericDim(128), new NumericDim(128))),
            new TensorType(
                FLOAT_32, asList(new NumericDim(8), new NumericDim(10), new NumericDim(10))),
            new TensorType(
                FLOAT_32, asList(new NumericDim(8), new NumericDim(100), new NumericDim(100))),
            new TensorType(
                FLOAT_32, asList(new NumericDim(16), new NumericDim(100), new NumericDim(100)))));

    test(
        TestCorpusFixtures.NLPGNN_FULL_PROJECT_FILES,
        "nlpgnn/layers/transformer.py",
        "Transformer.call",
        "nlpgnn_full_proj",
        2,
        17,
        expected);
  }
}
