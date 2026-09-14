package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * An array built from builtin conversions whose reader two callers give different conversions
 * (wala/ML#925), on {@code tf2_test_array_builtin_conversions.py}: the {@code dtype} travels two
 * hops, from the reader into the helper that builds the array, as at the site a census found in the
 * field. A one-hop form does not reach the defect: the dtype argument resolves per calling context
 * and the value walks never meet the builtin. In the two-hop form the builtin's type token itself
 * reaches the dtype value reader, whose allocation-site lookup threw; the worklist resolver caught
 * the exception and floored the whole array query to unknown, and nothing above fine level said so.
 * The shape value reader's identical arm is reached by the co-flow witness below, not by the
 * conversions.
 *
 * <p>The lookup now declines and the readers contribute nothing for such a key, so the query is
 * evaluated rather than floored. The values read here are the same before and after (unknown shape
 * either way, since the rows come from splitting strings of unknown count): what this test pins is
 * the census's count of query evaluations floored by a caught exception, 2 on the previous engine
 * and 0 now, with the values as the equivalence half. The runtime shapes are {@code (2, 3)} int64
 * and {@code (2, 2)} float64; neither array's shape is recovered, and the float array's dtype reads
 * unknown where the int array's reads int64, an asymmetry recorded here and not explained by this
 * unit.
 */
public class TestArrayBuiltinConversions extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_array_builtin_conversions.py";

  private static final String FILE_PROPERTY = "wala.ml.callgraph.census.file";

  private static final Set<TensorType> UNKNOWN_SHAPE_INT64 = Set.of(new TensorType(INT_64, null));

  private static final Set<TensorType> UNKNOWN_SHAPE_UNKNOWN =
      Set.of(new TensorType(UNKNOWN, null));

  @Test
  public void testIntConversionsReadWithoutAFlooredQuery()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    String[] census = analyseWithCensus("consume_ints", UNKNOWN_SHAPE_INT64);
    // The witness: on the previous engine this analysis floored 2 query evaluations by a caught
    // exception (the extractor's throw on the builtin token at each value reader); now none.
    assertEquals("query evaluations floored by a caught exception: " + census[7], "0", census[7]);
  }

  @Test
  public void testFloatConversionsReadWithoutAFlooredQuery()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    String[] census = analyseWithCensus("consume_floats", UNKNOWN_SHAPE_UNKNOWN);
    assertEquals("query evaluations floored by a caught exception: " + census[7], "0", census[7]);
  }

  /**
   * The shape reader's decline, which the conversions above never reach (their token meets only the
   * dtype reader): a builtin's type token sharing a container with a tensor is one member of the
   * operand's points-to set at {@code tf.identity(item)}, beside the tensor's allocation. On the
   * previous engine the extractor's throw at each value reader floored the whole result to {@code ?
   * of unknown} (measured); a decline contributes nothing for the token and keeps the tensor's
   * shape and dtype.
   */
  @Test
  public void testTokenBesideATensorKeepsTheTensorShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    String[] census = analyseWithCensus("consume_identity", Set.of(TENSOR_2_3_FLOAT32));
    assertEquals("query evaluations floored by a caught exception: " + census[7], "0", census[7]);
  }

  private String[] analyseWithCensus(String function, Set<TensorType> expected)
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Path file = Files.createTempDirectory("array-conversions").resolve("census.csv");
    String old = System.getProperty(FILE_PROPERTY);
    System.setProperty(FILE_PROPERTY, file.toString());
    try {
      test(FIXTURE, function, 1, 1, Map.of(2, expected));
    } finally {
      if (old == null) System.clearProperty(FILE_PROPERTY);
      else System.setProperty(FILE_PROPERTY, old);
    }
    List<String> lines = Files.readAllLines(file);
    assertEquals("one census line", 1, lines.size());
    return lines.get(0).split(",");
  }
}
