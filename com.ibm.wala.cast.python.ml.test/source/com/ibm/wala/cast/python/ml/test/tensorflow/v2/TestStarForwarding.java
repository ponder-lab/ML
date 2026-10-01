package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a wrapper whose formals are {@code *args} and {@code **kwargs} forwards every
 * positional and keyword argument to the function it calls (<a
 * href="https://github.com/wala/ML/issues/991">wala/ML#991</a>).
 */
public class TestStarForwarding extends AbstractTensorTest {

  private static final String FILE = "tf2_test_star_forwarding.py";

  private static final TensorType T2 = TensorType.of(FLOAT_32, 2);

  private static final TensorType T3 = TensorType.of(FLOAT_32, 3);

  private static final TensorType T4 = TensorType.of(FLOAT_32, 4);

  /** Two positional arguments reach their own parameters. */
  @Test
  public void testTwoPositionals() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "two_positionals", 2, 2, Map.of(2, Set.of(T2), 3, Set.of(T3)));
  }

  /** A keyword argument reaches the parameter it names through {@code **kwargs}. */
  @Test
  public void testKeyword() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "keyword", 2, 2, Map.of(2, Set.of(T2), 3, Set.of(T4)));
  }

  /** Positional and keyword arguments together. */
  @Test
  public void testMixed() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "mixed", 3, 3, Map.of(2, Set.of(T2), 3, Set.of(T3), 4, Set.of(T4)));
  }

  /** A method's {@code *args} packs the arguments of a call through an instance. */
  @Test
  public void testMethodVarargs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "method_target", 2, 2, Map.of(2, Set.of(T2), 3, Set.of(T3)));
  }

  /** A starred argument to a method called through an instance unpacks into its parameters. */
  @Test
  public void testStarredMethodCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "forwarded_target", 2, 2, Map.of(2, Set.of(T2), 3, Set.of(T3)));
  }

  /** A {@code **} argument that is a dict literal at the call binds the formal it names. */
  @Test
  public void testDoubleStarLiteral() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "literal_kw", 2, 2, Map.of(2, Set.of(T2), 3, Set.of(T4)));
  }

  /** As {@link #testDoubleStarLiteral()}, with the dict literal in a local. */
  @Test
  public void testDoubleStarLocal() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "local_kw", 2, 2, Map.of(2, Set.of(T2), 3, Set.of(T4)));
  }

  /** A {@code **} argument to a method called through an instance binds the formal it names. */
  @Test
  public void testDoubleStarMethod() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "Holder.take", 2, 2, Map.of(3, Set.of(T2), 4, Set.of(T4)));
  }

  /** A starred argument to a constructor unpacks into its initializer's parameters. */
  @Test
  public void testStarredConstructor()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "Built.__init__", 2, 2, Map.of(3, Set.of(T2), 4, Set.of(T3)));
  }

  /** A starred literal's elements past the named formals are packed into {@code *rest}. */
  @Test
  public void testStarredSpillsIntoVarargs()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "spill_sink", 1, 1, Map.of(2, Set.of(T3)));
  }

  /** A starred list built by {@code append} packs its elements of unknown index. */
  @Test
  public void testStarredAppendedList()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "appended_sink", 1, 1, Map.of(2, Set.of(T2)));
  }

  /** A keyword naming no formal is collected into {@code **kw}. */
  @Test
  public void testKeywordCollected() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "kw_sink", 1, 1, Map.of(2, Set.of(T4)));
  }

  /** A {@code **} dict a call computes binds {@code **kw} whole. */
  @Test
  public void testDoubleStarComputedIntoKwargs()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "dict_kw_sink", 1, 1, Map.of(2, Set.of(T4)));
  }
}
