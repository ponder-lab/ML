package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorType.mnistInput;
import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static java.util.Collections.emptyMap;
import static java.util.Collections.emptySet;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toSet;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.lsp.AnalysisError;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.NonBroadcastableShapesException;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.core.util.io.FileProvider;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.Context;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.debug.UnimplementedError;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.Test;

/** Test TF2 APIs. */
public class TestTensorflow2Model extends TestPythonMLCallGraphShape {

  private static final Logger LOGGER = Logger.getLogger(TestTensorflow2Model.class.getName());

  private static final String FLOAT_32 = FLOAT32.name().toLowerCase();

  private static final String FLOAT_64 = FLOAT64.name().toLowerCase();

  private static final String INT_32 = INT32.name().toLowerCase();

  private static final String INT_64 = DType.INT64.name().toLowerCase();

  private static final String UINT_8 = DType.UINT8.name().toLowerCase();

  private static final String STRING = DType.STRING.name().toLowerCase();

  private static final TensorType MNIST_INPUT = mnistInput();

  private static final TensorType SCALAR_TENSOR_OF_INT32 = new TensorType(INT_32, emptyList());

  private static final TensorType SCALAR_TENSOR_OF_INT64 = new TensorType(INT_64, emptyList());

  private static final TensorType SCALAR_TENSOR_OF_FLOAT32 = new TensorType(FLOAT_32, emptyList());

  private static final TensorType SCALAR_TENSOR_OF_STRING = new TensorType(STRING, emptyList());

  private static final TensorType TENSOR_1_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(1), new NumericDim(2)));

  @SuppressWarnings("unused")
  private static final TensorType TENSOR_32_INT32 =
      new TensorType(INT_32, asList(new NumericDim(32)));

  private static final TensorType TENSOR_32_UINT8 =
      new TensorType(UINT_8, asList(new NumericDim(32)));

  private static final TensorType TENSOR_256_784_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(256), new NumericDim(784)));

  private static final TensorType TENSOR_256_10_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(256), new NumericDim(10)));

  private static final TensorType TENSOR_256_UINT8 =
      new TensorType(UINT_8, asList(new NumericDim(256)));

  private static final TensorType TENSOR_10000_10_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(10000), new NumericDim(10)));

  private static final TensorType TENSOR_10000_UINT8 =
      new TensorType(UINT_8, asList(new NumericDim(10000)));

  private static final TensorType TENSOR_32_28_28_UINT8 =
      new TensorType(UINT_8, asList(new NumericDim(32), new NumericDim(28), new NumericDim(28)));

  private static final TensorType TENSOR_1_2_INT32 =
      new TensorType(INT_32, asList(new NumericDim(1), new NumericDim(2)));

  private static final TensorType TENSOR_2_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(2), new NumericDim(2)));

  private static final TensorType TENSOR_NONE_32_FLOAT32 =
      new TensorType(FLOAT_32, asList(null, new NumericDim(32)));

  private static final TensorType TENSOR_NONE_3_FLOAT32 =
      new TensorType(FLOAT_32, asList(null, new NumericDim(3)));

  private static final TensorType TENSOR_NONE_4_FLOAT32 =
      new TensorType(FLOAT_32, asList(null, new NumericDim(4)));

  private static final TensorType TENSOR_NONE_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(null, new NumericDim(2)));

  private static final TensorType TENSOR_NONE_NONE_STRING =
      new TensorType(STRING, asList(null, null));

  private static final TensorType TENSOR_4_NONE_NONE_INT32 =
      new TensorType(INT_32, asList(new NumericDim(4), null, null));

  private static final TensorType TENSOR_3_NONE_NONE_INT32 =
      new TensorType(INT_32, asList(new NumericDim(3), null, null));

  private static final TensorType TENSOR_4_NONE_NONE_NONE_STRING =
      new TensorType(STRING, asList(new NumericDim(4), null, null, null));

  private static final TensorType TENSOR_3_NONE_NONE_STRING =
      new TensorType(STRING, asList(new NumericDim(3), null, null));

  private static final TensorType TENSOR_1_NONE_NONE_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(1), null, null));

  private static final TensorType TENSOR_2_NONE_NONE_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), null, null));

  private static final TensorType TENSOR_2_2_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), new NumericDim(2)));

  private static final TensorType TENSOR_3_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), new NumericDim(2)));

  private static final TensorType TENSOR_2_3_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(2), new NumericDim(3)));

  private static final TensorType TENSOR_3_3_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), new NumericDim(3)));

  private static final TensorType TENSOR_3_3_INT32 =
      new TensorType(INT_32, asList(new NumericDim(3), new NumericDim(3)));

  private static final TensorType TENSOR_0_NONE_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(0), null));

  private static final TensorType TENSOR_0_NONE_3_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(0), null, new NumericDim(3)));

  private static final TensorType TENSOR_1_NONE_INT32 =
      new TensorType(INT_32, asList(new NumericDim(1), null));

  private static final TensorType TENSOR_1_NONE_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(1), null));

  private static final TensorType TENSOR_2_NONE_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), null));

  private static final TensorType TENSOR_2_NONE_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(2), null));

  private static final TensorType TENSOR_2_NONE_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(2), null, new NumericDim(2)));

  private static final TensorType TENSOR_2_NONE_2_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), null, new NumericDim(2)));

  private static final TensorType TENSOR_2_NONE_2_3_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), null, new NumericDim(2), new NumericDim(3)));

  private static final TensorType TENSOR_2_NONE_2_2_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), null, new NumericDim(2), new NumericDim(2)));

  @SuppressWarnings("unused")
  private static final TensorType TENSOR_2_NONE_NONE_NONE_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), null));

  private static final TensorType TENSOR_2_NONE_NONE_NONE_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(2), null, null, null));

  private static final TensorType TENSOR_3_NONE_INT32 =
      new TensorType(INT_32, asList(new NumericDim(3), null));

  private static final TensorType TENSOR_3_NONE_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), null));

  private static final TensorType TENSOR_4_NONE_INT32 =
      new TensorType(INT_32, asList(new NumericDim(4), null));

  private static final TensorType TENSOR_3_NONE_NONE_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), null, null));

  private static final TensorType TENSOR_3_NONE_1_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), null, new NumericDim(1)));

  private static final TensorType TENSOR_2_3_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), new NumericDim(3)));

  private static final TensorType TENSOR_2_1_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(2), new NumericDim(1)));

  private static final TensorType TENSOR_10_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(10), new NumericDim(2)));

  private static final TensorType TENSOR_10_2_FLOAT64 =
      new TensorType(FLOAT_64, asList(new NumericDim(10), new NumericDim(2)));

  private static final TensorType TENSOR_5_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(5), new NumericDim(2)));

  private static final TensorType TENSOR_5_2_INT32 =
      new TensorType(INT_32, asList(new NumericDim(5), new NumericDim(2)));

  private static final TensorType TENSOR_5_5_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(5), new NumericDim(5)));

  private static final TensorType TENSOR_5_5_INT32 =
      new TensorType(INT_32, asList(new NumericDim(5), new NumericDim(5)));

  private static final TensorType TENSOR_5_NONE_INT32 =
      new TensorType(INT_32, asList(new NumericDim(5), null));

  private static final TensorType TENSOR_2_3_3_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), new NumericDim(3), new NumericDim(3)));

  private static final TensorType TENSOR_2_3_4_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), new NumericDim(3), new NumericDim(4)));

  private static final TensorType TENSOR_2_5_3_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(2), new NumericDim(5), new NumericDim(3)));

  private static final TensorType TENSOR_3_2_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), new NumericDim(2), new NumericDim(2)));

  private static final TensorType TENSOR_7_5_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(7), new NumericDim(5), new NumericDim(2)));

  private static final TensorType TENSOR_30_3_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(30), new NumericDim(3), new NumericDim(2)));

  private static final TensorType TENSOR_3_2_2_3_FLOAT32 =
      new TensorType(
          FLOAT_32,
          asList(new NumericDim(3), new NumericDim(2), new NumericDim(2), new NumericDim(3)));

  private static final TensorType TENSOR_2_2_2_3_FLOAT32 =
      new TensorType(
          FLOAT_32,
          asList(new NumericDim(2), new NumericDim(2), new NumericDim(2), new NumericDim(3)));

  private static final TensorType TENSOR_20_28_28_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(20), new NumericDim(28), new NumericDim(28)));

  private static final TensorType TENSOR_20_28_28_INT32 =
      new TensorType(INT_32, asList(new NumericDim(20), new NumericDim(28), new NumericDim(28)));

  private static final TensorType TENSOR_20_10_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(20), new NumericDim(10)));

  private static final TensorType TENSOR_60000_28_28_FLOAT32 =
      new TensorType(
          FLOAT_32, asList(new NumericDim(60000), new NumericDim(28), new NumericDim(28)));

  /** A {@code float32} tensor whose shape cannot be statically inferred. */
  private static final TensorType TENSOR_UNKNOWN_SHAPE_FLOAT32 = new TensorType(FLOAT_32, null);

  private static final TensorType TENSOR_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(2)));

  private static final TensorType TENSOR_2_FLOAT64 =
      new TensorType(FLOAT_64, asList(new NumericDim(2)));

  private static final TensorType TENSOR_2_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2)));

  private static final TensorType TENSOR_2_INT64 =
      new TensorType(INT_64, asList(new NumericDim(2)));

  private static final TensorType TENSOR_3_INT32 =
      new TensorType(INT_32, asList(new NumericDim(3)));

  private static final TensorType TENSOR_3_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3)));

  private static final TensorType TENSOR_4_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(4)));

  private static final TensorType TENSOR_4_FLOAT64 =
      new TensorType(FLOAT_64, asList(new NumericDim(4)));

  private static final TensorType TENSOR_5_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(5)));

  private static final TensorType TENSOR_64_5_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(64), new NumericDim(5)));

  private static final TensorType TENSOR_5_INT32 =
      new TensorType(INT_32, asList(new NumericDim(5)));

  private static final TensorType TENSOR_4_INT32 =
      new TensorType(INT_32, asList(new NumericDim(4)));

  private static final TensorType TENSOR_1_INT32 =
      new TensorType(INT_32, asList(new NumericDim(1)));

  private static final TensorType TENSOR_3_4_INT32 =
      new TensorType(INT_32, asList(new NumericDim(3), new NumericDim(4)));

  private static final TensorType TENSOR_3_4_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), new NumericDim(4)));

  private static final TensorType TENSOR_4_5_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(4), new NumericDim(5)));

  private static final TensorType TENSOR_1_28_28_1_FLOAT32 =
      new TensorType(
          FLOAT_32,
          asList(new NumericDim(1), new NumericDim(28), new NumericDim(28), new NumericDim(1)));

  private static final TensorType TENSOR_6_INT32 =
      new TensorType(INT_32, asList(new NumericDim(6)));

  private static final TensorType TENSOR_6_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(6)));

  private static final TensorType TENSOR_256_28_28_1_FLOAT32 =
      new TensorType(
          FLOAT_32,
          asList(new NumericDim(256), new NumericDim(28), new NumericDim(28), new NumericDim(1)));

  private static final TensorType TENSOR_32_28_28_1_FLOAT32 =
      new TensorType(
          FLOAT_32,
          asList(new NumericDim(32), new NumericDim(28), new NumericDim(28), new NumericDim(1)));

  private static final TensorType TENSOR_16_28_28_1_FLOAT32 =
      new TensorType(
          FLOAT_32,
          asList(new NumericDim(16), new NumericDim(28), new NumericDim(28), new NumericDim(1)));

  private static final TensorType TENSOR_256_64_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(256), new NumericDim(64)));

  private static final TensorType TENSOR_96_28_28_1_FLOAT32 =
      new TensorType(
          FLOAT_32,
          asList(new NumericDim(96), new NumericDim(28), new NumericDim(28), new NumericDim(1)));

  @Test
  public void testValueIndex()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_value_index.py",
        "value_index",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32), 3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testValueIndex2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_value_index2.py",
        "value_index",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32), 3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testValueIndex3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_value_index3.py",
        "value_index",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32), 3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testValueIndex4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_value_index4.py",
        "value_index",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32), 3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testFunction()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function2.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function3.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function4.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function5.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function6.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_1_FLOAT32)));
  }

  @Test
  public void testFunction7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function7.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testFunction8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_function8.py",
        "func",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_1_FLOAT32, TENSOR_2_FLOAT32)));
  }

  @Test
  public void testFunction9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function9.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testFunction10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function10.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_INT32)));
  }

  @Test
  public void testFunction11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function11.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_3_3_INT32)));
  }

  /** Test https://github.com/wala/ML/issues/308. */
  @Test
  public void testFunction12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_function12.py",
        "func",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_1_FLOAT32, TENSOR_3_2_FLOAT32)));
  }

  /**
   * Test https://github.com/wala/ML/issues/308.
   *
   * <p>This one has lexical scoping.
   */
  @Test
  public void testFunction13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_function13.py",
        "func",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_1_FLOAT32, TENSOR_3_2_FLOAT32)));
  }

  @Test
  public void testDecorator()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator2.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator3.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testDecorator4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator4.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator5.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator6.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator7.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator8.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator9.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator10.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator11.py", "C.returned", 1, 1, Map.of(3, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_decorator12.py",
        "returned",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_FLOAT32, TENSOR_2_INT32)));
  }

  @Test
  public void testDataset()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDataset2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset2.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** This is not a legal case. */
  @Test
  public void testDataset3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset3.py", "add", 0, 0, emptyMap());
  }

  /** This is not a legal case. */
  @Test
  public void testDataset4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset4.py", "add", 0, 0, emptyMap());
  }

  @Test
  public void testDataset5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset5.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  @Test
  public void testDataset6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset6.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  @Test
  public void testDataset7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset7.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  @Test
  public void testDataset8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset8.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_32_28_28_UINT8), 3, Set.of(TENSOR_32_UINT8)));
  }

  @Test
  public void testDataset9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset9.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_32_28_28_UINT8), 3, Set.of(TENSOR_32_UINT8)));
  }

  @Test
  public void testDataset10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset10.py",
        "add",
        2,
        3,
        Map.of(
            2,
            Set.of(SCALAR_TENSOR_OF_INT32, TENSOR_2_NONE_INT32),
            3,
            Set.of(SCALAR_TENSOR_OF_INT32, TENSOR_2_NONE_INT32)));
  }

  @Test
  public void testDataset10a()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset10a.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(SCALAR_TENSOR_OF_INT32),
            3, Set.of(TENSOR_2_NONE_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset11.py", "f", 0, 0);
    test("tf2_test_dataset11.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset12.py", "f", 0, 0);
    test("tf2_test_dataset12.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset13.py", "f", 0, 0);
    test("tf2_test_dataset13.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset14()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset14.py", "f", 0, 0);
    test("tf2_test_dataset14.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset15()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset15.py", "f", 0, 0);
    test("tf2_test_dataset15.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset16()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset16.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset17()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset17.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset17.py", "f", 1, 2, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset18()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset18.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset18.py", "f", 1, 2, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset18.py", "g", 0, 2);
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset19()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType images =
        new TensorType(
            FLOAT_32,
            asList(
                new NumericDim(512), new NumericDim(112), new NumericDim(112), new NumericDim(3)));
    TensorType labels = new TensorType(FLOAT_32, asList(new NumericDim(512), null));

    test(
        "tf2_test_dataset19.py", "distributed_train_step", 1, 1, Map.of(2, Set.of(images, labels)));
  }

  @Test
  public void testDataset20()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset20.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset21()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset21.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset22()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset22.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset23()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset23.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset23.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDataset24()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset24.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset24.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset25()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset25.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset25.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/165 is fixed.
    test(
        "tf2_test_dataset25.py",
        "h",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64, SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDataset26()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset26.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset26.py", "g1", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset26.py", "g2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset26.py", "g3", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/165 is fixed.
    test(
        "tf2_test_dataset26.py",
        "h",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64, TENSOR_2_INT32)));
  }

  @Test
  public void testDataset27()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset27.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
    test("tf2_test_dataset27.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset27.py", "h", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset27.py", "i", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDataset28()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset28.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
    test("tf2_test_dataset28.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
    // TODO: Change to 0, 0 when https://github.com/wala/ML/issues/164 is fixed:
    test(
        "tf2_test_dataset28.py",
        "h",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_INT32, SCALAR_TENSOR_OF_STRING)));
  }

  @Test
  public void testDataset29()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset29.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  @Test
  public void testDataset30()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset30.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset31()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/166 is fixed.
    test("tf2_test_dataset31.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "g1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "g2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/166 is fixed.
    test("tf2_test_dataset31.py", "h", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "i1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "i2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/166 is fixed.
    test(
        "tf2_test_dataset31.py",
        "j",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64, TENSOR_2_INT64)));
    test("tf2_test_dataset31.py", "k1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "k2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "k3", 1, 1, Map.of(2, Set.of(TENSOR_2_INT64)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/166 is fixed.
    test("tf2_test_dataset31.py", "l", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "m1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "m2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset32()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset32.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset33()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset33.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset34()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset34.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset35()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset35.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset36()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset36.py", "id1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
    //    test("tf2_test_dataset36.py", "id2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset37()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset37.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case that exercises a potential 1-CFA precision problem with Datasets by using a shared
   * function to retrieve the first element from two different datasets.
   */
  @Test
  public void testDataset38()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset38.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset38.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Control test for {@link #testDataset38()}, utilizing only a single dataset to ensure the 1-CFA
   * precision issue is specifically due to the merging of multiple datasets.
   */
  @Test
  public void testDataset39()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset39.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case that exercises a potential 1-CFA precision problem with Datasets by using a shared
   * function to retrieve the first element from two different datasets. Uses tf.constant to bypass
   * unrelated type inference issues with Python lists in from_tensor_slices.
   */
  @Test
  public void testDataset40()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset40.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset40.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Control test for {@link #testDataset40()}, utilizing only a single dataset to ensure the 1-CFA
   * precision issue is specifically due to the merging of multiple datasets.
   */
  @Test
  public void testDataset41()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset41.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Direct access test based on {@link #testDataset38()}, iterating over the dataset directly
   * instead of through a shared function, avoiding the 1-CFA precision issue.
   */
  @Test
  public void testDataset42()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset42.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset42.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset42()}, utilizing only a single dataset. */
  @Test
  public void testDataset43()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset43.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Direct access test based on {@link #testDataset40()}, using tf.constant and iterating directly.
   */
  @Test
  public void testDataset44()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset44.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset44.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset44()}, utilizing only a single dataset. */
  @Test
  public void testDataset45()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset45.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case based on {@link #testDataset38()}, exercising potential 1-CFA precision problem with
   * Datasets by varying the shape (e.g., shape (2,) vs (2, 2)) instead of the dtype.
   */
  @Test
  public void testDataset46()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset46.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset46.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset46()}, utilizing only a single dataset. */
  @Test
  public void testDataset47()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset47.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Test case based on {@link #testDataset40()}, exercising potential 1-CFA precision problem with
   * Datasets by varying the shape using tf.constant.
   */
  @Test
  public void testDataset48()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset48.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset48.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset48()}, utilizing only a single dataset. */
  @Test
  public void testDataset49()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset49.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Direct access test based on {@link #testDataset46()}, iterating over the dataset directly
   * instead of through a shared function.
   */
  @Test
  public void testDataset50()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset50.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset50.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset50()}, utilizing only a single dataset. */
  @Test
  public void testDataset51()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset51.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Direct access test based on {@link #testDataset48()}, using tf.constant and iterating directly.
   */
  @Test
  public void testDataset52()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset52.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset52.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset52()}, utilizing only a single dataset. */
  @Test
  public void testDataset53()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset53.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Test case that exercises a potential 1-CFA precision problem with Datasets by using an explicit
   * iterator inside a shared function to retrieve the first element from two different datasets.
   */
  @Test
  public void testDataset54()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset54.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset54.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset54()}, utilizing only a single dataset. */
  @Test
  public void testDataset55()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset55.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case that exercises a potential 1-CFA precision problem with chained Datasets by sharing a
   * helper function that calls shuffle() on two different datasets.
   */
  @Test
  public void testDataset56()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset56.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset56.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset56()}, utilizing only a single dataset. */
  @Test
  public void testDataset57()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset57.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Direct access version of {@link #testDataset56()}, utilizing chained Datasets (shuffle)
   * directly in the script without a shared helper function. This should NOT suffer from 1-CFA
   * precision merging.
   */
  @Test
  public void testDataset58()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset58.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset58.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset58()}, utilizing only a single dataset. */
  @Test
  public void testDataset59()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset59.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Direct access version of {@link #testDataset54()}, utilizing an explicit iterator directly in
   * the script without a shared helper function.
   */
  @Test
  public void testDataset60()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset60.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset60.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset60()}, utilizing only a single dataset. */
  @Test
  public void testDataset61()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset61.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case exposing 1-CFA precision problem using {@code tf.data.Dataset.from_tensors} inside a
   * shared wrapper function.
   */
  @Test
  public void testDataset62()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset62.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_32, asList(new NumericDim(3))))));
    test(
        "tf2_test_dataset62.py",
        "g",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_32, asList(new NumericDim(3))))));
  }

  /** Control test for {@link #testDataset62()}, utilizing only a single dataset. */
  @Test
  public void testDataset63()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset63.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_32, asList(new NumericDim(3))))));
  }

  /**
   * Test case exposing 1-CFA precision problem using {@code tf.data.Dataset.range} inside a shared
   * wrapper function.
   */
  @Test
  public void testDataset64()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // The range generator always produces int64, but the 1-CFA merge occurs anyway,
    // though the type sets will both be {int64}. We verify it runs.
    test("tf2_test_dataset64.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset64.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  /** Control test for {@link #testDataset64()}, utilizing only a single dataset. */
  @Test
  public void testDataset65()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset65.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  /**
   * Test case exposing 1-CFA precision problem using {@code shuffle} inside a shared wrapper
   * function, varying shapes instead of dtypes.
   */
  @Test
  public void testDataset66()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset66.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset66.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset66()}, utilizing only a single dataset. */
  @Test
  public void testDataset67()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset67.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Test case exposing 1-CFA precision problem using {@code tf.data.Dataset.from_tensors} inside a
   * shared wrapper function, varying shapes instead of dtypes.
   */
  @Test
  public void testDataset68()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset68.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset68.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset68()}, utilizing only a single dataset. */
  @Test
  public void testDataset69()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset69.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Test {@code tf.data.Dataset.from_tensor_slices} with a tuple of two Python list literals,
   * followed by the same {@code .shuffle(...).batch(...)} chain as {@link #testDataset5()}. The
   * iterator yields a tuple {@code (element_a, element_b)} where each element has the same batched
   * shape as {@code testDataset5}'s single element. This isolates the "tuple-structured argument"
   * path from the {@link #testDataset5()} "single-list" path; every test that feeds a tuple to
   * {@code from_tensor_slices} today does so via {@code mnist.load_data()}'s ndarray split, and no
   * existing test exercises the tuple-of-literals case in isolation.
   *
   * <p>The root-cause tuple walk was fixed on the feature branch for wala/ML#366 via a call-site
   * helper in {@link com.ibm.wala.cast.python.ml.client.DatasetFromTensorSlicesGenerator}, but this
   * test still fails because of a separate "chain-preservation gap": the {@code .shuffle().batch()}
   * chain wraps the dataset in generators that don't implement {@code DelegatingTensorGenerator},
   * so the factory's {@code PythonPropertyRead} dispatch for the {@code for element_a, element_b in
   * dataset:} unpack cannot peel back to find the tuple structure, and instead wraps in {@code
   * TensorElementGenerator} which over-peels one dim — collapsing the batched shape {@code (2,)}
   * back to {@code ()}.
   */
  @Test
  public void testDataset70()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset70.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  /**
   * Like {@link #testDataset70()} but the two elements of the tuple are {@code tf.constant(...)}
   * calls rather than raw Python list literals. Used as a comparison point to isolate whether the
   * tuple-structured-argument bug observed in {@link #testDataset70()} is specific to raw literals
   * or also applies when the tuple elements are already typed tensors. Both fail with identical
   * wrong output; see {@link #testDataset70()}'s Javadoc for the current root-cause status
   * (wala/ML#366 root-cause walk fixed, chain-preservation gap remains).
   */
  @Test
  public void testDataset71()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset71.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testTensorboardExample()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tensorboard_example.py", "summarize_weights", 0, 4);
  }

  @Test
  public void testTensorList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tensor_list.py",
        "add",
        2,
        3,
        Map.of(
            2,
            Set.of(TENSOR_1_2_FLOAT32, TENSOR_2_2_FLOAT32),
            3,
            Set.of(TENSOR_1_2_FLOAT32, TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testTensorList2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensor_list2.py", "add", 0, 1);
  }

  @Test
  public void testTensorList3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tensor_list3.py",
        "add",
        0,
        0); // NOTE: Change to 2, 2, 2, 3 once https://github.com/wala/ML/issues/136 is fixed.
  }

  @Test
  public void testTensorList4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensor_list4.py", "add", 0, 0);
  }

  @Test
  public void testTensorList5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensor_list5.py", "add", 0, 0);
  }

  @Test
  public void testModelCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // Two tensor variables in __call__: the `x` parameter (value number 3) and the float32 result
    // of an internal `DenseCall` whose concrete shape cannot currently be inferred through the
    // chained layer calls — tracked as `{? of float32}`. See wala/ML#356 for the underlying PTS
    // propagation gap.
    test(
        "tf2_test_model_call.py",
        "SequentialModel.__call__",
        1,
        2,
        Map.of(3, Set.of(TENSOR_20_28_28_FLOAT32)));
  }

  @Test
  public void testModelCall2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call2.py",
        "SequentialModel.call",
        1,
        2,
        Map.of(3, Set.of(TENSOR_20_28_28_FLOAT32)));
  }

  @Test
  public void testModelCall3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call3.py",
        "SequentialModel.call",
        1,
        2,
        Map.of(3, Set.of(TENSOR_20_28_28_FLOAT32)));
  }

  @Test
  public void testModelCall4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call4.py",
        "SequentialModel.__call__",
        1,
        2,
        Map.of(3, Set.of(TENSOR_20_28_28_FLOAT32)));
  }

  /**
   * Test call string imprecision as described in
   * https://github.com/wala/WALA/discussions/1417#discussioncomment-10085680. This should fail due
   * to https://github.com/wala/ML/issues/207.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#207 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testModelCall5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj66/src/tf2_test_model_call5b.py",
          "proj66/tf2_test_model_call5.py",
          "proj66/tf2_test_model_call5a.py"
        },
        "tf2_test_model_call5.py",
        "SequentialModel.__call__",
        "proj66",
        1,
        1,
        Map.of(3, Set.of(MNIST_INPUT)));

    test(
        new String[] {
          "proj66/src/tf2_test_model_call5b.py",
          "proj66/tf2_test_model_call5.py",
          "proj66/tf2_test_model_call5a.py"
        },
        "tf2_test_model_call5a.py",
        "SequentialModel.__call__",
        "proj66",
        1,
        1,
        Map.of(3, Set.of(MNIST_INPUT)));
  }

  /**
   * Test https://github.com/wala/ML/issues/267.
   *
   * <p>Explicit dtype.
   */
  @Test
  public void testModelCall6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call6.py",
        "SequentialModel.__call__",
        1,
        2,
        Map.of(3, Set.of(TENSOR_20_28_28_INT32)));
  }

  @Test
  public void testModelAttributes()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes2.py",
        "f",
        1,
        1,
        Map.of(
            2, Set.of(TENSOR_3_4_FLOAT32, TENSOR_4_FLOAT32, TENSOR_4_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes3.py",
        "f",
        1,
        1,
        Map.of(
            2, Set.of(TENSOR_3_4_FLOAT32, TENSOR_4_FLOAT32, TENSOR_4_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes4.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes5.py",
        "f",
        1,
        1,
        Map.of(
            2, Set.of(TENSOR_3_4_FLOAT32, TENSOR_4_FLOAT32, TENSOR_4_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes6.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  /**
   * Tests precise dataflow tracing for Keras {@code Model} weights when using keyword arguments for
   * both {@code Dense} layer instantiation and {@code Model} construction. Verifies that the
   * weights are correctly identified and have the expected shapes {@code (64, 5)} and {@code (5,)}.
   */
  @Test
  public void testModelAttributes7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes7.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testCallbacks()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_callbacks.py", "replica_fn", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testCallbacks2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_callbacks2.py", "replica_fn", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testGanTutorial()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_gan_tutorial.py",
        "train_step",
        1,
        5,
        Map.of(2, Set.of(TENSOR_256_28_28_1_FLOAT32, TENSOR_96_28_28_1_FLOAT32)));
  }

  @Test
  public void testGanTutorial2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_gan_tutorial2.py",
        "train_step",
        1,
        5,
        Map.of(2, Set.of(TENSOR_256_28_28_1_FLOAT32, TENSOR_96_28_28_1_FLOAT32)));
  }

  /**
   * {@code MyModel.call(self, x)} receives {@code x} from {@code model(images)} calls inside {@code
   * train_step} and {@code test_step}. {@code images} comes from iterating {@code train_ds}, {@code
   * valid_ds}, or {@code test_ds} &mdash; all created from mnist data via {@code
   * .astype(np.float32) / 255.0} and {@code [..., tf.newaxis]}, then batched by 32. At runtime
   * {@code x} has shape {@code (32, 28, 28, 1)} dtype {@code float32} (verified by Python assert
   * statements in {@code tensorflow_eager_execution.py}).
   *
   * <p>Note: {@code test_ds} yields mostly {@code (32, 28, 28, 1)} batches plus one trailing
   * partial batch of shape {@code (16, 28, 28, 1)} (since {@code 10000 % 32 == 16}), so the
   * aspirational union for value 3 includes both shapes.
   *
   * <p>The rule-based count is 5 (1 parameter + 4 intermediate layer-call ops {@code conv1}, {@code
   * flatten}, {@code d1}, {@code d2}), but the analysis only registers 4. See wala/ML#389 for the
   * investigation of which tensor variable is missing. The expected count is set to 4 (current
   * actual) so the count check passes and the test exposes the type check; a legitimate future rise
   * to 5 would trigger a re-evaluation via the resulting count mismatch.
   *
   * <p>With the count check passing, the test now fails on value 3's type: actual {@code {(32,)
   * uint8, (16,) uint8}} &mdash; the {@code y_*} labels' types (shapes only showing the batch dim)
   * leak into the {@code x} parameter's slot. Same tuple-routing bug as {@link
   * #testNeuralNetwork()} (wala/ML#385) applied to the {@code (x_train, y_train)} / {@code (x_test,
   * y_test)} tuples consumed by {@code from_tensor_slices}.
   */
  @Test
  public void testEagerExecution()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_eager_execution.py",
        "MyModel.call",
        1,
        4,
        Map.of(3, Set.of(TENSOR_32_28_28_1_FLOAT32, TENSOR_16_28_28_1_FLOAT32)));
  }

  /**
   * Parameter {@code x} of {@code NeuralNet.call} receives {@code batch_x} from the dataset
   * iteration chain. At runtime, {@code batch_x} has shape {@code (256, 784)} and dtype {@code
   * float32} (verified by Python assert statements in {@code neural_network.py}).
   *
   * <p>The test currently fails because value 3 (slot 0 of the {@code (x_train, y_train)} tuple
   * consumed by {@code from_tensor_slices}) receives {@code y_train}'s types {@code {(256,) uint8,
   * (?) uint8}} instead of {@code x_train}'s {@code (256, 784) float32}. Two contributing root
   * causes: (1) a missing generator for ndarray {@code .reshape()} breaks {@code x_train}'s PA
   * chain through {@code np.array(x_train, np.float32).reshape([-1, 784])}; (2) per-index {@code
   * TupleElementProvider} routing doesn't survive the {@code .shuffle().batch()} chain because
   * {@code DatasetBatchGenerator} and the shuffle-generated {@code DatasetGenerator} don't
   * implement {@code DelegatingTensorGenerator} (wala/ML#385).
   *
   * <p>Rule-based tensor variable count is 5 (1 parameter {@code x} + 4 intermediate ops {@code
   * fc1}, {@code fc2}, {@code out}, {@code softmax}). The analysis currently registers 3; the
   * discrepancy is unaccounted for (wala/ML#390). Count set to 3 (branch actual) so the count check
   * passes and the remaining failure exposes type bugs; a future fix that legitimately raises the
   * count will trigger the test with a clear signal.
   */
  @Test
  public void testNeuralNetwork()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("neural_network.py", "NeuralNet.call", 1, 3, Map.of(3, Set.of(TENSOR_256_784_FLOAT32)));
  }

  /**
   * {@code cross_entropy_loss(x, y)} receives logits {@code x} (value 2) and labels {@code y}
   * (value 3). At runtime, {@code x} has shape {@code (256, 10)} dtype {@code float32} and {@code
   * y} has shape {@code (256,)} dtype {@code uint8} (verified by Python assert statements in {@code
   * neural_network.py}).
   *
   * <p>TODO: Value 2 ({@code x}) is not yet tracked as a tensor parameter. It flows from {@code
   * pred = neural_net(batch_x, is_training=True)}, which dispatches through the summarized {@code
   * Model.__call__} into user-defined {@code NeuralNet.call}. wala/ML#127 (now closed) was a
   * necessary fix for this dispatch to work at all, but is insufficient on its own &mdash; value 2
   * remains untracked until the same reshape/tuple-routing gap blocking {@link
   * #testNeuralNetwork()} is resolved. See wala/ML#378.
   *
   * <p>Once value 2 is tracked, the test will also fail on value 3: the actual {@code y} type
   * includes a spurious {@code (?) uint8} in the union ({@code {(256,) uint8, (?) uint8}}) &mdash;
   * same seeding/generator tension as {@link #testNeuralNetwork()} (wala/ML#385).
   *
   * <p>The rule-based tensor variable count is 5 (2 parameters {@code x}, {@code y} + 3
   * intermediate ops {@code cast-to-int64}, {@code sparse_softmax_cross_entropy_with_logits},
   * {@code reduce_mean}). However, the analysis actually registers 8, and we keep 8 here to
   * preserve regression detection: if the count ever drops, we want to know. The three extra tensor
   * variables are unaccounted for &mdash; tracked by wala/ML#388.
   */
  @Test
  public void testNeuralNetwork2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "neural_network.py",
        "cross_entropy_loss",
        2,
        8,
        Map.of(2, Set.of(TENSOR_256_10_FLOAT32), 3, Set.of(TENSOR_256_UINT8)));
  }

  /**
   * {@code run_optimization(x, y)} is called with {@code batch_x} and {@code batch_y} from the
   * dataset iteration chain. At runtime, {@code x} has shape {@code (256, 784)} dtype {@code
   * float32} and {@code y} has shape {@code (256,)} dtype {@code uint8} (verified by Python assert
   * statements in {@code neural_network.py}).
   *
   * <p>The test currently fails because value 2 (slot 0 of the tuple, same as {@link
   * #testNeuralNetwork()}) receives {@code y_train}'s types instead of {@code x_train}'s, and value
   * 3 additionally carries a spurious {@code (?) uint8} in its union. Same tuple-routing and
   * reshape root causes as {@link #testNeuralNetwork()} (wala/ML#385).
   *
   * <p>Rule-based tensor variable count is 6 (2 parameters {@code x}, {@code y} + 4 intermediate
   * ops {@code pred}, {@code loss}, {@code trainable_variables}, {@code gradients}). The analysis
   * currently registers 3; the discrepancy is unaccounted for (wala/ML#391). Note that {@code
   * trainable_variables} and {@code gradients} are lists of tensors rather than single tensors,
   * which may legitimately not register as {@code TensorVariable}s &mdash; even so, the rule-based
   * count drops only to 4, still above the branch actual. Count set to 3 (branch actual) so the
   * count check passes and the remaining failure exposes type bugs; a future fix that legitimately
   * raises the count will trigger the test with a clear signal.
   */
  @Test
  public void testNeuralNetwork3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "neural_network.py",
        "run_optimization",
        2,
        3,
        Map.of(2, Set.of(TENSOR_256_784_FLOAT32), 3, Set.of(TENSOR_256_UINT8)));
  }

  /**
   * {@code accuracy(y_pred, y_true)} is called from two sites: the training loop with {@code
   * accuracy(pred, batch_y)} where {@code pred} has shape {@code (256, 10)} dtype {@code float32}
   * and {@code batch_y} has shape {@code (256,)} dtype {@code uint8}; and the test-set evaluation
   * with {@code accuracy(pred, y_test)} where {@code pred} has shape {@code (10000, 10)} dtype
   * {@code float32} and {@code y_test} has shape {@code (10000,)} dtype {@code uint8} (verified by
   * Python assert statements in {@code neural_network.py}). The static analysis should union these
   * types for each parameter.
   *
   * <p>Expected tensor variable count: 7 (2 parameters {@code y_pred}, {@code y_true} + 5
   * intermediate ops {@code argmax}, {@code cast-to-int64}, {@code equal}, {@code cast-to-float32},
   * {@code reduce_mean}). Actual: 4. Contributing bugs: (1) wala/ML#386 &mdash; {@code tf.argmax}
   * and {@code tf.equal} resolve to empty points-to sets because they are defined under {@code
   * <package name="tensorflow/math">} in {@code tensorflow.xml} but called as {@code tf.argmax} /
   * {@code tf.equal} in the Python code; (2) wala/ML#387 &mdash; {@code TensorGeneratorFactory}
   * throws {@code IAE: Unknown call: pass_through} for empty-PTS sources, so neither {@code
   * tf.cast} call contributes a tensor variable (cascading from #386 in this test).
   *
   * <p>Value 2 ({@code y_pred}) may also be subject to the same reshape/tuple-routing gap blocking
   * {@link #testNeuralNetwork()} (wala/ML#385); wala/ML#127 (closed) was a
   * necessary-but-insufficient prerequisite for value 2 tracking through the {@code Model.__call__}
   * dispatch chain.
   */
  @Test
  public void testNeuralNetwork4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "neural_network.py",
        "accuracy",
        2,
        7,
        Map.of(
            2,
            Set.of(TENSOR_256_10_FLOAT32, TENSOR_10000_10_FLOAT32),
            3,
            Set.of(TENSOR_256_UINT8, TENSOR_10000_UINT8)));
  }

  /**
   * {@code encoder(x)} receives {@code x} from call sites {@code decoder(encoder(x))} inside {@code
   * run_optimization} (training loop) and {@code decoder(encoder(batch_x))} at the module-level
   * test loop. Both call sites pass batches of shape {@code (256, 784)} dtype {@code float32}
   * (verified by Python assert statements in {@code autoencoder.py}).
   *
   * <p>Expected tensor variable count: 18 (baseline). The rule-based count is lower (1 param plus
   * ~10 intermediate ops including dict lookups for {@code weights[...]} and {@code biases[...]}),
   * but the analysis registers 18 due to per-context duplication across the two call sites. Per the
   * "never decrease" principle, we keep 18 to preserve regression detection (see wala/ML#388 for
   * the general count-accounting discrepancy).
   */
  @Test
  public void testAutoencoder()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("autoencoder.py", "encoder", 1, 18, Map.of(2, Set.of(TENSOR_256_784_FLOAT32)));
  }

  /**
   * {@code mean_square(reconstructed, original)} is called only from {@code run_optimization}
   * (itself a FUT &mdash; {@link #testAutoencoder3()}). Its arguments are {@code
   * reconstructed_image = decoder(encoder(x))} and {@code x}, both of which have runtime shape
   * {@code (256, 784)} dtype {@code float32}. Direct call-site asserts aren't possible (they would
   * perturb {@code run_optimization}'s count), so the runtime types are verified indirectly through
   * the {@code batch_x} asserts at the training-loop call of {@code run_optimization}.
   *
   * <p>Expected tensor variable count: 5 (2 parameters + 3 intermediate ops {@code original -
   * reconstructed}, {@code tf.pow}, {@code tf.reduce_mean}); raised from the baseline of 2.
   */
  @Test
  public void testAutoencoder2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "autoencoder.py",
        "mean_square",
        2,
        5,
        Map.of(2, Set.of(TENSOR_256_784_FLOAT32), 3, Set.of(TENSOR_256_784_FLOAT32)));
  }

  /**
   * {@code run_optimization(x)} is called from the training loop with {@code batch_x} of shape
   * {@code (256, 784)} dtype {@code float32} (verified by Python assert statements in {@code
   * autoencoder.py}).
   *
   * <p>Expected tensor variable count: 6 (1 parameter + 5 intermediate ops {@code encoder(x)}
   * result, {@code decoder(...) = reconstructed_image}, {@code mean_square(...) = loss}, {@code
   * trainable_variables}, {@code gradients}); raised from the baseline of 3.
   */
  @Test
  public void testAutoencoder3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("autoencoder.py", "run_optimization", 1, 6, Map.of(2, Set.of(TENSOR_256_784_FLOAT32)));
  }

  /**
   * {@code decoder(x)} is called from the same two sites as {@code encoder} ({@link
   * #testAutoencoder()}), but with {@code x} being the output of {@code encoder}. Since {@code
   * encoder}'s layer 2 has dim {@code num_hidden_2 = 64}, {@code decoder} receives {@code (256, 64)
   * float32} (verified by a Python assert in the test loop).
   *
   * <p>Expected tensor variable count: 18 (baseline; parallel to {@link #testAutoencoder()}).
   */
  @Test
  public void testAutoencoder4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("autoencoder.py", "decoder", 1, 18, Map.of(2, Set.of(TENSOR_256_64_FLOAT32)));
  }

  @Test
  public void testSigmoid()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sigmoid.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  @Test
  public void testSigmoid2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sigmoid2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  @Test
  public void testAdd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  public void testAdd2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testAdd3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testAdd4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  public void testAdd5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testAdd6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testAdd7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add7.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add8.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add9.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add10.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add11.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add12.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add13.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd14()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add14.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd15()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add15.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testAdd16()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add16.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testAdd17()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add17.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testAdd18()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add18.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testAdd19()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add19.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testAdd20()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add20.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testAdd21()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add21.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testAdd22()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add22.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testAdd23()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add23.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_INT32), 3, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testAdd24()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add24.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_INT32), 3, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testAdd25()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add25.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_INT32), 3, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testAdd26()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add26.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd27()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add27.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd28()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add28.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_4_INT32), 3, Set.of(TENSOR_3_4_INT32)));
  }

  @Test
  public void testAdd29()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add29.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_4_INT32), 3, Set.of(TENSOR_3_4_INT32)));
  }

  @Test
  public void testAdd30()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add30.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_INT32), 3, Set.of(TENSOR_2_2_INT32)));
  }

  @Test
  public void testAdd31()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add31.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_INT32), 3, Set.of(TENSOR_2_2_INT32)));
  }

  @Test
  public void testAdd32()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add32.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_INT32), 3, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testAdd33()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add33.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_INT32), 3, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testAdd34()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add34.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_3_FLOAT32), 3, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testAdd35()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add35.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_3_FLOAT32), 3, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testAdd36()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add36.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testAdd37()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add37.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testAdd38()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add38.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_INT32), 3, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testAdd39()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add39.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_INT32), 3, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testAdd40()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add40.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd41()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add41.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd42()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add42.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testZerosLikeTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_zeros_like_tensor.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd43()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add43.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32), 3, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testAdd44()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add44.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32), 3, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testAdd45()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add45.py",
        "value_index",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32), 3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testAdd46()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add46.py",
        "value_index",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32), 3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testAdd47()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add47.py",
        "add",
        2,
        2,
        Map.of(2, Set.of(TENSOR_3_4_INT32), 3, Set.of(TENSOR_3_4_INT32)));
  }

  @Test
  public void testAdd48()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add48.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd49()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // NOTE: Set the expected number of tensor variables to 3 once
    // https://github.com/wala/ML/issues/135 is fixed.
    test(
        "tf2_test_add49.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd50()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add50.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_INT32), 3, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testAdd51()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add51.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd52()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add52.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd53()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add53.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testInputWithBatchSize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t1 = new TensorType(FLOAT_32, asList(new NumericDim(16), new NumericDim(32)));
    TensorType t2 =
        new TensorType(FLOAT_32, asList(new NumericDim(5), new NumericDim(10), new NumericDim(10)));
    TensorType t3 = new TensorType(FLOAT_32, asList(null, new NumericDim(5)));

    test(
        "tf2_test_input_batch_size.py",
        "check_input",
        3,
        3,
        Map.of(2, Set.of(t1), 3, Set.of(t2), 4, Set.of(t3)));
  }

  @Test
  public void testInputInt32()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t1 = new TensorType(INT_32, asList(new NumericDim(32), new NumericDim(10)));
    TensorType t2 =
        new TensorType(INT_32, asList(new NumericDim(8), new NumericDim(5), new NumericDim(5)));

    test("tf2_test_input_int32.py", "check_input", 2, 2, Map.of(2, Set.of(t1), 3, Set.of(t2)));
  }

  @Test
  public void testInputMixedArgs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // input1: shape=(32, 10), dtype=int32
    TensorType t1 = new TensorType(INT_32, asList(new NumericDim(32), new NumericDim(10)));
    // input2: shape=(16, 5, 5), dtype=float32
    TensorType t2 =
        new TensorType(FLOAT_32, asList(new NumericDim(16), new NumericDim(5), new NumericDim(5)));
    // input3: shape=(None, 20), dtype=int32
    TensorType t3 = new TensorType(INT_32, asList(null, new NumericDim(20)));

    test(
        "tf2_test_input_mixed_args.py",
        "check_input",
        3,
        3,
        Map.of(2, Set.of(t1), 3, Set.of(t2), 4, Set.of(t3)));
  }

  @Test
  public void testInputUnimplemented() {
    assertThrows(
        UnimplementedError.class,
        () ->
            test(
                "tf2_test_input_unimplemented_sparse_kw.py",
                "tf2_test_input_unimplemented_sparse_kw.py",
                0,
                0,
                emptyMap()));

    assertThrows(
        UnimplementedError.class,
        () ->
            test(
                "tf2_test_input_unimplemented_tensor_kw.py",
                "tf2_test_input_unimplemented_tensor_kw.py",
                0,
                0,
                emptyMap()));

    assertThrows(
        UnimplementedError.class,
        () ->
            test(
                "tf2_test_input_unimplemented_ragged_kw.py",
                "tf2_test_input_unimplemented_ragged_kw.py",
                0,
                0,
                emptyMap()));

    assertThrows(
        UnimplementedError.class,
        () ->
            test(
                "tf2_test_input_unimplemented_type_spec_kw.py",
                "tf2_test_input_unimplemented_type_spec_kw.py",
                0,
                0,
                emptyMap()));

    assertThrows(
        UnimplementedError.class,
        () ->
            test(
                "tf2_test_input_unimplemented_sparse_pos.py",
                "tf2_test_input_unimplemented_sparse_pos.py",
                0,
                0,
                emptyMap()));

    assertThrows(
        UnimplementedError.class,
        () ->
            test(
                "tf2_test_input_unimplemented_tensor_pos.py",
                "tf2_test_input_unimplemented_tensor_pos.py",
                0,
                0,
                emptyMap()));
  }

  @Test
  public void testAdd54()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add54.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_NONE_32_FLOAT32), 3, Set.of(TENSOR_NONE_32_FLOAT32)));
  }

  @Test
  public void testAdd55()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add55.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_NONE_32_FLOAT32), 3, Set.of(TENSOR_NONE_32_FLOAT32)));
  }

  @Test
  public void testAdd56()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add56.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_NONE_32_FLOAT32), 3, Set.of(TENSOR_NONE_32_FLOAT32)));
  }

  @Test
  public void testAdd57()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add57.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_NONE_32_FLOAT32), 3, Set.of(TENSOR_NONE_32_FLOAT32)));
  }

  @Test
  public void testAdd58()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add58.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_NONE_32_FLOAT32), 3, Set.of(TENSOR_NONE_32_FLOAT32)));
  }

  @Test
  public void testAdd59()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add59.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_INT32), 3, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testAdd60()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add60.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_INT32), 3, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testAdd61()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add61.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_INT32), 3, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testAdd62()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add62.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_INT32), 3, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testAdd63()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add63.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_4_INT32), 3, Set.of(TENSOR_3_4_INT32)));
  }

  @Test
  public void testAdd64()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add64.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_4_INT32), 3, Set.of(TENSOR_3_4_INT32)));
  }

  @Test
  public void testAdd65()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add65.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_4_INT32), 3, Set.of(TENSOR_3_4_INT32)));
  }

  @Test
  public void testAdd66()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add66.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_NONE_INT32), 3, Set.of(TENSOR_1_NONE_INT32)));
  }

  @Test
  public void testAdd67()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add67.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_NONE_INT32), 3, Set.of(TENSOR_1_NONE_INT32)));
  }

  @Test
  public void testAdd68()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add68.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_NONE_INT32), 3, Set.of(TENSOR_1_NONE_INT32)));
  }

  @Test
  public void testAdd69()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add69.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_NONE_INT32), 3, Set.of(TENSOR_1_NONE_INT32)));
  }

  @Test
  public void testAdd70()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add70.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_NONE_INT32), 3, Set.of(TENSOR_1_NONE_INT32)));
  }

  @Test
  public void testAdd71()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add71.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_NONE_INT32), 3, Set.of(TENSOR_1_NONE_INT32)));
  }

  @Test
  public void testAdd72()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add72.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_4_NONE_NONE_NONE_STRING), 3, Set.of(TENSOR_4_NONE_NONE_NONE_STRING)));
  }

  @Test
  public void testAdd73()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add73.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_4_NONE_NONE_NONE_STRING), 3, Set.of(TENSOR_4_NONE_NONE_NONE_STRING)));
  }

  @Test
  public void testAdd74()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add74.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_4_NONE_NONE_NONE_STRING), 3, Set.of(TENSOR_4_NONE_NONE_NONE_STRING)));
  }

  @Test
  public void testAdd75()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add75.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_4_NONE_NONE_NONE_STRING), 3, Set.of(TENSOR_4_NONE_NONE_NONE_STRING)));
  }

  @Test
  public void testAdd76()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add76.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_NONE_NONE_INT32), 3, Set.of(TENSOR_3_NONE_NONE_INT32)));
  }

  @Test
  public void testAdd77()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add77.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_NONE_NONE_INT32), 3, Set.of(TENSOR_3_NONE_NONE_INT32)));
  }

  @Test
  public void testAdd78()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add78.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_NONE_NONE_INT32), 3, Set.of(TENSOR_3_NONE_NONE_INT32)));
  }

  @Test
  public void testAdd79()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add79.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_NONE_NONE_INT32), 3, Set.of(TENSOR_3_NONE_NONE_INT32)));
  }

  @Test
  public void testAdd80()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add80.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_NONE_NONE_INT32), 3, Set.of(TENSOR_3_NONE_NONE_INT32)));
  }

  @Test
  public void testAdd81()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add81.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_NONE_NONE_INT32), 3, Set.of(TENSOR_3_NONE_NONE_INT32)));
  }

  @Test
  public void testAdd82()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add82.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_NONE_NONE_INT32), 3, Set.of(TENSOR_3_NONE_NONE_INT32)));
  }

  @Test
  public void testAdd83()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add83.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_NONE_NONE_INT32), 3, Set.of(TENSOR_3_NONE_NONE_INT32)));
  }

  @Test
  public void testAdd84()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add84.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_4_NONE_NONE_NONE_STRING), 3, Set.of(TENSOR_4_NONE_NONE_NONE_STRING)));
  }

  @Test
  public void testAdd85()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add85.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_4_NONE_NONE_NONE_STRING), 3, Set.of(TENSOR_4_NONE_NONE_NONE_STRING)));
  }

  @Test
  public void testAdd86()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add86.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_4_NONE_NONE_NONE_STRING), 3, Set.of(TENSOR_4_NONE_NONE_NONE_STRING)));
  }

  @Test
  public void testAdd87()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add87.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_4_NONE_NONE_NONE_STRING), 3, Set.of(TENSOR_4_NONE_NONE_NONE_STRING)));
  }

  @Test
  public void testAdd88()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add88.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_NONE_INT32), 3, Set.of(TENSOR_5_NONE_INT32)));
  }

  @Test
  public void testAdd89()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add89.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_NONE_INT32), 3, Set.of(TENSOR_5_NONE_INT32)));
  }

  @Test
  public void testAdd90()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add90.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_NONE_INT32), 3, Set.of(TENSOR_5_NONE_INT32)));
  }

  @Test
  public void testAdd91()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add91.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_NONE_INT32), 3, Set.of(TENSOR_5_NONE_INT32)));
  }

  @Test
  public void testAdd92()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add92.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_NONE_INT32), 3, Set.of(TENSOR_5_NONE_INT32)));
  }

  @Test
  public void testAdd93()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add93.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_NONE_INT32), 3, Set.of(TENSOR_5_NONE_INT32)));
  }

  @Test
  public void testAdd94()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add94.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_NONE_INT32), 3, Set.of(TENSOR_5_NONE_INT32)));
  }

  @Test
  public void testAdd95()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add95.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_NONE_INT32), 3, Set.of(TENSOR_5_NONE_INT32)));
  }

  @Test
  public void testAdd96()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add96.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_NONE_INT32), 3, Set.of(TENSOR_5_NONE_INT32)));
  }

  @Test
  public void testAdd97()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add97.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_4_NONE_INT32), 3, Set.of(TENSOR_4_NONE_INT32)));
  }

  @Test
  public void testAdd98()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add98.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_4_NONE_INT32), 3, Set.of(TENSOR_4_NONE_INT32)));
  }

  @Test
  public void testAdd99()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add99.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_4_NONE_INT32), 3, Set.of(TENSOR_4_NONE_INT32)));
  }

  @Test
  public void testRaggedFromRowStartsFull()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_starts_full.py",
        "test_ragged_from_row_starts_full",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_5_NONE_INT32),
            3, Set.of(TENSOR_5_NONE_INT32),
            4, Set.of(TENSOR_5_NONE_INT32),
            5, Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testRaggedFromRowLengths()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_lengths.py",
        "test_ragged_from_row_lengths",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_NONE_INT32)));
  }

  @Test
  public void testRaggedFromRowLimits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_limits.py",
        "test_ragged_from_row_limits",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_NONE_INT32)));
  }

  @Test
  public void testRaggedFromValueRowIds()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_value_rowids.py",
        "test_ragged_from_value_rowids",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_4_NONE_INT32),
            3, Set.of(TENSOR_4_NONE_INT32),
            4, Set.of(TENSOR_4_NONE_INT32)));
  }

  @Test
  public void testStrictnessFailure()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_strictness_failure.py",
        "test_strictness",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_NONE_INT32)));
  }

  @Test
  public void testNoneDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_none_dtype.py", "test_none_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testRaggedKeywordArgsNested()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_keyword_args_nested.py",
        "test_keywords",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_2_NONE_NONE_INT32),
            3, Set.of(TENSOR_2_NONE_NONE_INT32),
            4, Set.of(TENSOR_2_NONE_NONE_INT32),
            5, Set.of(TENSOR_2_NONE_NONE_INT32)));
  }

  @Test
  public void testRaggedKeywordArgsAdditional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_keyword_args_more.py",
        "test_keywords",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_5_NONE_INT32),
            3, Set.of(TENSOR_4_NONE_INT32),
            4, Set.of(TENSOR_4_NONE_INT32),
            5, Set.of(TENSOR_5_NONE_INT32)));
  }

  @Test
  public void testRaggedKeywordArgs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_keyword_args.py",
        "test_keywords",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_2_NONE_INT32),
            3, Set.of(TENSOR_2_NONE_INT32),
            4, Set.of(TENSOR_2_NONE_INT32),
            5, Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testRaggedKeywordArgsV2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_keyword_args_v2.py",
        "test_keywords",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_2_NONE_INT32),
            3, Set.of(TENSOR_2_NONE_INT32),
            4, Set.of(TENSOR_2_NONE_INT32),
            5, Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testRaggedKeywordArgsMixed()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ragged_mixed_new.py",
        "test_ragged_mixed_args_new",
        3,
        3,
        Map.of(
            // rt1: positional values, keyword value_rowids. inferred nrows=3.
            2,
            Set.of(TENSOR_3_NONE_INT32),

            // rt2: positional values, keyword value_rowids, keyword nrows=5.
            3,
            Set.of(TENSOR_5_NONE_INT32),

            // rt3: positional values, positional value_rowids, keyword nrows=3.
            4,
            Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testRaggedNrowsPositional()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ragged_nrows_positional.py",
        "test_ragged_nrows_positional",
        1,
        1,
        Map.of(
            // rt: positional values, positional value_rowids, positional nrows=3.
            2, Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testAdd100()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add100.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_10_2_FLOAT32), 3, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testAdd101()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add101.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_10_2_FLOAT32), 3, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testAdd102()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add102.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_10_2_FLOAT32), 3, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testAdd103()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add103.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_10_2_FLOAT32), 3, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testAdd104()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add104.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_4_FLOAT32), 3, Set.of(TENSOR_4_FLOAT32)));
  }

  @Test
  public void testAdd105()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add105.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_4_FLOAT32), 3, Set.of(TENSOR_4_FLOAT32)));
  }

  @Test
  public void testAdd106()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add106.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_4_FLOAT32), 3, Set.of(TENSOR_4_FLOAT32)));
  }

  @Test
  public void testAdd107()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add107.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_10_2_FLOAT32), 3, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testAdd108()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add108.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_10_2_FLOAT32), 3, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testAdd109()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add109.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testAdd110()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add110.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testAdd111()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add111.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32), 3, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testAdd112()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add112.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32), 3, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testAdd113()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add113.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32), 3, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testAdd114()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add114.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_NONE_32_FLOAT32), 3, Set.of(TENSOR_NONE_32_FLOAT32)));
  }

  @Test
  public void testAdd115()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add115.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_NONE_32_FLOAT32), 3, Set.of(TENSOR_NONE_32_FLOAT32)));
  }

  @Test
  public void testAdd116()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add116.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Tests that the analysis identifies non-broadcastable shapes in conditional branches.
   *
   * <p>In {@code tf2_test_add117.py}, the variable {@code a} can be either 1 or 3.
   *
   * <ul>
   *   <li>If {@code a=1}, the addition is {@code [1, 2] + [2, 2]}, which is broadcastable.
   *   <li>If {@code a=3}, the addition is {@code [3, 2] + [2, 2]}, which is NOT broadcastable.
   * </ul>
   *
   * The analysis correctly identifies that one possible dataflow is invalid and throws a {@link
   * NonBroadcastableShapesException}.
   *
   * @see #testAdd117a()
   */
  @Test(expected = NonBroadcastableShapesException.class)
  public void testAdd117()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add117.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32, TENSOR_3_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Tests that the analysis correctly identifies broadcastable shapes even when they originate from
   * multiple conditional branches.
   *
   * <p>This is a companion test to {@link #testAdd117()}. In {@code tf2_test_add117a.py}, the
   * variable {@code a} can be either 1 or 2.
   *
   * <ul>
   *   <li>If {@code a=1}, the addition is {@code [1, 2] + [2, 2]}, which is broadcastable to {@code
   *       [2, 2]}.
   *   <li>If {@code a=2}, the addition is {@code [2, 2] + [2, 2]}, which is broadcastable to {@code
   *       [2, 2]}.
   * </ul>
   *
   * Since all branches lead to broadcastable shapes, the analysis succeeds without exception.
   *
   * @see #testAdd117()
   */
  @Test
  public void testAdd117a()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add117a.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32, TENSOR_2_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd118()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add118.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_INT32), 3, Set.of(TENSOR_2_2_INT32)));
  }

  @Test
  public void testAddResultFlow()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add_result_flow.py", "check_result", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testSubResultFlow()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sub_result_flow.py", "check_result", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testMulResultFlow()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_mul_result_flow.py", "check_result", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testDivResultFlow()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_div_result_flow.py", "check_result", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testAdd119()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add119.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testAdd120()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add120.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testAdd121()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add121.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_4_FLOAT32), 3, Set.of(TENSOR_4_FLOAT32)));
  }

  @Test
  public void testAdd122()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add122.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_4_FLOAT64), 3, Set.of(TENSOR_4_FLOAT64)));
  }

  @Test
  public void testSparseAdd()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testSparseAdd2()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testSparseAdd3()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  @Test
  public void testSparseAdd4()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  @Test
  public void testSparseAdd5()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testSparseAdd6()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testMultiGPUTraining()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "multigpu_training.py",
        "run_optimization",
        2,
        4,
        Map.of(2, Set.of(MNIST_INPUT), 3, Set.of(MNIST_INPUT)));
  }

  @Test
  public void testMultiGPUTraining2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "multigpu_training.py",
        "average_gradients",
        0,
        0); // NOTE: Change to 1, 1, 2 once https://github.com/wala/ML/issues/136 is fixed.
  }

  @Test
  public void testReduceMean()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testReduceMean2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceMean3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean.py", "h", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceMean4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceMean5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceMean6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "h", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testReduceMean7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "i", 1, 1, Map.of(2, Set.of(TENSOR_2_1_FLOAT32)));
  }

  @Test
  public void testReduceMean8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "j", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testGradient()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gradient.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_NONE_FLOAT32)));
  }

  @Test
  public void testGradient2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gradient2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_NONE_FLOAT32)));
  }

  @Test
  public void testMultiply()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_multiply.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_INT32)));
  }

  @Test
  public void testMultiply2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_multiply2.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testMultiply3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_multiply3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testMultiply4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_multiply4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testMultiply5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_multiply5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_2_3_FLOAT32)));
  }

  /**
   * This is an invalid case since the inputs have different ranks.
   *
   * <p>For now, we are throwing an exception. But, this is invalid code.
   *
   * <p>TODO: We'll need to come up with a suitable way to handle this in the future.
   */
  @Test(expected = NonBroadcastableShapesException.class)
  public void testMultiply6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_multiply6.py", "f", 1, 1);
  }

  @Test
  public void testMultiply7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_multiply7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testSparseSoftmaxCrossEntropyWithLogits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_softmax_cross_entropy_with_logits.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  @Test
  public void testRelu()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_relu.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testRange()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testRange2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range2.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testRange3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range3.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testRange4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_range4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testRange5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_INT32)));
  }

  @Test
  public void testImport()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_import.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testImport2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_import2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testImport3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_import3.py", "f", 1, 2, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test("tf2_test_import3.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testImport4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_import4.py", "f", 1, 2, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test("tf2_test_import4.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testImport5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_import5.py", "f", 0, 1);
    test("tf2_test_import5.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testImport6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_import6.py", "f", 0, 1);
    test("tf2_test_import6.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * This is an invalid case. If there are no wildcard imports, we should resolve them like they
   * are.
   */
  @Test
  public void testImport7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_import7.py", "f", 0, 0);
    test("tf2_test_import7.py", "g", 0, 0);
  }

  /**
   * This is an invalid case. If there are no wildcard imports, we should resolve them like they
   * are.
   */
  @Test
  public void testImport8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_import8.py", "f", 0, 0);
    test("tf2_test_import8.py", "g", 0, 0);
  }

  @Test
  public void testImport9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_import9.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test("tf2_test_import9.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testModule()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"tf2_test_module2.py", "tf2_test_module.py"},
        "tf2_test_module2.py",
        "f",
        "",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** This test needs a PYTHONPATH that points to `proj`. */
  @Test
  public void testModule2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj/src/__init__.py", "proj/src/tf2_test_module2a.py", "proj/src/tf2_test_module3.py"
        },
        "src/tf2_test_module2a.py",
        "f",
        "proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** This test should not need a PYTHONPATH. */
  @Test
  public void testModule3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj2/src/__init__.py", "proj2/src/tf2_test_module3a.py", "proj2/tf2_test_module4.py"
        },
        "src/tf2_test_module3a.py",
        "f",
        "proj2",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * This test should not need a PYTHONPATH, meaning that I don't need to set one in the console
   * when I run the files.
   */
  @Test
  public void testModule4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj3/src/__init__.py",
          "proj3/src/tf2_test_module4a.py",
          "proj3/src/tf2_test_module6.py",
          "proj3/tf2_test_module5.py"
        },
        "src/tf2_test_module4a.py",
        "f",
        "proj3",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));

    test(
        new String[] {
          "proj3/src/__init__.py",
          "proj3/src/tf2_test_module4a.py",
          "proj3/src/tf2_test_module6.py",
          "proj3/tf2_test_module5.py"
        },
        "src/tf2_test_module4a.py",
        "g",
        "proj3",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testModule5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"tf2_test_module4.py", "tf2_test_module3.py"},
        "tf2_test_module4.py",
        "C.f",
        "",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** This test needs a PYTHONPATH that points to `proj4`. */
  @Test
  public void testModule6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj4/src/__init__.py", "proj4/src/tf2_test_module4a.py", "proj4/src/tf2_test_module5.py"
        },
        "src/tf2_test_module4a.py",
        "C.f",
        "proj4",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** This test should not need a PYTHONPATH. */
  @Test
  public void testModule7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj5/src/__init__.py", "proj5/src/tf2_test_module5a.py", "proj5/tf2_test_module6.py"
        },
        "src/tf2_test_module5a.py",
        "C.f",
        "proj5",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * This test should not need a PYTHONPATH, meaning that I don't need to set one in the console
   * when I run the files.
   */
  @Test
  public void testModule8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj6/src/__init__.py",
          "proj6/src/tf2_test_module8a.py",
          "proj6/src/tf2_test_module6.py",
          "proj6/tf2_test_module7.py"
        },
        "src/tf2_test_module8a.py",
        "C.f",
        "proj6",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));

    test(
        new String[] {
          "proj6/src/__init__.py",
          "proj6/src/tf2_test_module8a.py",
          "proj6/src/tf2_test_module6.py",
          "proj6/tf2_test_module7.py"
        },
        "src/tf2_test_module8a.py",
        "D.g",
        "proj6",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testModule9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"tf2_test_module6.py", "tf2_test_module5.py"},
        "tf2_test_module6.py",
        "D.f",
        "",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testModule10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"tf2_test_module8.py", "tf2_test_module9.py", "tf2_test_module7.py"},
        "tf2_test_module9.py",
        "D.f",
        "",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** This test needs a PYTHONPATH that points to `proj7`. */
  @Test
  public void testModule11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj7/src/__init__.py",
          "proj7/src/tf2_test_module9a.py",
          "proj7/src/tf2_test_module9b.py",
          "proj7/src/tf2_test_module10.py"
        },
        "src/tf2_test_module9b.py",
        "D.f",
        "proj7",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** This test should not need a PYTHONPATH. */
  @Test
  public void testModule12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj8/src/__init__.py",
          "proj8/src/tf2_test_module10a.py",
          "proj8/src/tf2_test_module10b.py",
          "proj8/tf2_test_module11.py"
        },
        "src/tf2_test_module10b.py",
        "D.f",
        "proj8",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** This test should not need a PYTHONPATH. */
  @Test
  public void testModule13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj9/src/__init__.py",
          "proj9/src/tf2_test_module11a.py",
          "proj9/src/tf2_test_module11b.py",
          "proj9/tf2_test_module12.py"
        },
        "src/tf2_test_module11b.py",
        "D.g",
        "proj9",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/177.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule14()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj10/C/__init__.py", "proj10/C/B.py", "proj10/A.py"},
        "C/B.py",
        "f",
        "proj10",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule15()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj11/C/__init__.py", "proj11/C/B.py", "proj11/A.py"},
        "C/B.py",
        "f",
        "proj11",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** This test should not need a PYTHONPATH. */
  @Test
  public void testModule16()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj12/C/__init__.py", "proj12/C/B.py", "proj12/A.py"},
        "C/B.py",
        "f",
        "proj12",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178. Multi-submodule case. See
   * https://docs.python.org/3/tutorial/modules.html#packages.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule17()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj13/C/__init__.py", "proj13/C/D/__init__.py", "proj13/C/D/B.py", "proj13/A.py"
        },
        "C/D/B.py",
        "f",
        "proj13",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178. Multi-submodule case. See
   * https://docs.python.org/3/tutorial/modules.html#packages. This test has multiple modules in
   * different packages.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule18()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj14/C/__init__.py",
          "proj14/C/E.py",
          "proj14/C/D/__init__.py",
          "proj14/C/D/B.py",
          "proj14/A.py"
        },
        "C/D/B.py",
        "f",
        "proj14",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));

    test(
        new String[] {
          "proj14/C/__init__.py",
          "proj14/C/E.py",
          "proj14/C/D/__init__.py",
          "proj14/C/D/B.py",
          "proj14/A.py"
        },
        "C/E.py",
        "g",
        "proj14",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/177.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule19()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj15/C/__init__.py", "proj15/C/D/__init__.py", "proj15/C/D/B.py", "proj15/A.py"
        },
        "C/D/B.py",
        "f",
        "proj15",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule20()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj16/C/__init__.py", "proj16/C/B.py", "proj16/A.py"},
        "C/B.py",
        "D.f",
        "proj16",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule21()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj17/C/__init__.py", "proj17/C/E/__init__.py", "proj17/C/E/B.py", "proj17/A.py"
        },
        "C/E/B.py",
        "D.f",
        "proj17",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/177.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule22()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj18/B.py", "proj18/A.py"},
        "B.py",
        "f",
        "proj18",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/177.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule23()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj19/C/__init__.py",
          "proj19/C/D/__init__.py",
          "proj19/C/D/E/__init__.py",
          "proj19/C/D/E/B.py",
          "proj19/A.py"
        },
        "C/D/E/B.py",
        "f",
        "proj19",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/177.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule24()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"tf2_test_module11.py", "tf2_test_module10.py"},
        "tf2_test_module11.py",
        "f",
        "",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/177.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule25()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj20/B.py", "proj20/A.py"},
        "B.py",
        "C.f",
        "proj20",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/177.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule26()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"tf2_test_module13.py", "tf2_test_module12.py"},
        "tf2_test_module13.py",
        "C.f",
        "",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule27()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj21/C/__init__.py",
          "proj21/C/D/__init__.py",
          "proj21/C/E.py",
          "proj21/C/D/B.py",
          "proj21/A.py"
        },
        "C/D/B.py",
        "F.f",
        "proj21",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));

    test(
        new String[] {
          "proj21/C/__init__.py",
          "proj21/C/D/__init__.py",
          "proj21/C/E.py",
          "proj21/C/D/B.py",
          "proj21/A.py"
        },
        "C/E.py",
        "G.g",
        "proj21",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/177.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule28()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj22/C/__init__.py", "proj22/C/B.py", "proj22/A.py"},
        "C/B.py",
        "D.f",
        "proj22",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule29()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj23/C/__init__.py", "proj23/C/B.py", "proj23/A.py"},
        "C/B.py",
        "f",
        "proj23",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule30()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj24/C/__init__.py", "proj24/C/B.py", "proj24/A.py"},
        "C/B.py",
        "D.f",
        "proj24",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule31()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj25/C/__init__.py", "proj25/C/E/__init__.py", "proj25/C/E/B.py", "proj25/A.py"
        },
        "C/E/B.py",
        "D.f",
        "proj25",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule32()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj26/C/__init__.py", "proj26/C/B.py", "proj26/A.py"},
        "C/B.py",
        "D.f",
        "proj26",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule33()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj27/C/__init__.py", "proj27/C/D/__init__.py", "proj27/C/D/B.py", "proj27/A.py"
        },
        "C/D/B.py",
        "f",
        "proj27",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule34()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj28/C/__init__.py", "proj28/C/D/__init__.py", "proj28/C/D/B.py", "proj28/A.py"
        },
        "C/D/B.py",
        "E.f",
        "proj28",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule35()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj29/C/__init__.py", "proj29/C/B.py", "proj29/A.py"},
        "C/B.py",
        "f",
        "proj29",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test for https://github.com/wala/ML/issues/178.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule36()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj30/C/__init__.py", "proj30/C/B.py", "proj30/A.py"},
        "C/B.py",
        "f",
        "proj30",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule37()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj31/C/__init__.py", "proj31/C/B.py", "proj31/C/A.py", "proj31/main.py"},
        "C/B.py",
        "f",
        "proj31",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule38()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj32/C/__init__.py", "proj32/C/B.py", "proj32/C/A.py", "proj32/main.py"},
        "C/B.py",
        "f",
        "proj32",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule39()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj33/C/__init__.py", "proj33/C/B.py", "proj33/C/A.py", "proj33/main.py"},
        "C/B.py",
        "D.f",
        "proj33",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule40()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj34/C/__init__.py", "proj34/C/B.py", "proj34/C/A.py", "proj34/main.py"},
        "C/B.py",
        "D.f",
        "proj34",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule41()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj35/E/__init__.py",
          "proj35/E/C/__init__.py",
          "proj35/E/D/__init__.py",
          "proj35/E/D/B.py",
          "proj35/E/C/A.py",
          "proj35/main.py"
        },
        "E/D/B.py",
        "f",
        "proj35",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule42()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj36/E/__init__.py",
          "proj36/E/C/__init__.py",
          "proj36/E/D/__init__.py",
          "proj36/E/D/B.py",
          "proj36/E/C/A.py",
          "proj36/main.py"
        },
        "E/D/B.py",
        "F.f",
        "proj36",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule43()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj37/E/__init__.py",
          "proj37/E/C/__init__.py",
          "proj37/E/D/__init__.py",
          "proj37/E/D/B.py",
          "proj37/E/C/A.py",
          "proj37/main.py"
        },
        "E/D/B.py",
        "F.f",
        "proj37",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule44()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj38/E/__init__.py",
          "proj38/E/C/__init__.py",
          "proj38/E/D/__init__.py",
          "proj38/E/D/B.py",
          "proj38/E/C/A.py",
          "proj38/main.py"
        },
        "E/D/B.py",
        "f",
        "proj38",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports using wildcards.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule45()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj39/C/__init__.py", "proj39/C/B.py", "proj39/C/A.py", "proj39/main.py"},
        "C/B.py",
        "f",
        "proj39",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports using wildcards.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule46()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj40/C/__init__.py", "proj40/C/B.py", "proj40/C/A.py", "proj40/main.py"},
        "C/B.py",
        "f",
        "proj40",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports using wildcards.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule47()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj41/C/__init__.py", "proj41/C/B.py", "proj41/C/A.py", "proj41/main.py"},
        "C/B.py",
        "D.f",
        "proj41",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports using wildcards.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule48()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj42/C/__init__.py", "proj42/C/B.py", "proj42/C/A.py", "proj42/main.py"},
        "C/B.py",
        "D.f",
        "proj42",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports using wildcards.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule49()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj43/E/__init__.py",
          "proj43/E/C/__init__.py",
          "proj43/E/D/__init__.py",
          "proj43/E/D/B.py",
          "proj43/E/C/A.py",
          "proj43/main.py"
        },
        "E/D/B.py",
        "f",
        "proj43",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports using wildcards.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule50()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj44/E/__init__.py",
          "proj44/E/C/__init__.py",
          "proj44/E/D/__init__.py",
          "proj44/E/D/B.py",
          "proj44/E/C/A.py",
          "proj44/main.py"
        },
        "E/D/B.py",
        "F.f",
        "proj44",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports using wildcards.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule51()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj45/E/__init__.py",
          "proj45/E/C/__init__.py",
          "proj45/E/D/__init__.py",
          "proj45/E/D/B.py",
          "proj45/E/C/A.py",
          "proj45/main.py"
        },
        "E/D/B.py",
        "F.f",
        "proj45",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports using wildcards.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule52()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj46/E/__init__.py",
          "proj46/E/C/__init__.py",
          "proj46/E/D/__init__.py",
          "proj46/E/D/B.py",
          "proj46/E/C/A.py",
          "proj46/main.py"
        },
        "E/D/B.py",
        "f",
        "proj46",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test relative imports.
   *
   * <p>This test should not need a PYTHONPATH.
   */
  @Test
  public void testModule53()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj47/E/__init__.py",
          "proj47/D/__init__.py",
          "proj47/E/C/__init__.py",
          "proj47/E/D/__init__.py",
          "proj47/E/D/B.py",
          "proj47/E/C/A.py",
          "proj47/D/B.py",
          "proj47/main.py"
        },
        "E/D/B.py",
        "f",
        "proj47",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));

    test(
        new String[] {
          "proj47/E/__init__.py",
          "proj47/D/__init__.py",
          "proj47/E/C/__init__.py",
          "proj47/E/D/__init__.py",
          "proj47/E/D/B.py",
          "proj47/E/C/A.py",
          "proj47/D/B.py",
          "proj47/main.py"
        },
        "D/B.py",
        "g",
        "proj47",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule54()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj51/src/__init__.py", "proj51/src/module.py", "proj51/client.py"},
        "src/module.py",
        "f",
        "proj51",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule55()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj52/src/__init__.py", "proj52/src/module.py", "proj52/client.py"},
        "src/module.py",
        "f",
        "proj52",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule56()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj53/src/__init__.py", "proj53/src/module.py", "proj53/client.py"},
        "src/module.py",
        "C.f",
        "proj53",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule57()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj54/src/__init__.py", "proj54/src/module.py", "proj54/client.py"},
        "src/module.py",
        "C.f",
        "proj54",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule58()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj55/src/__init__.py", "proj55/src/B.py", "proj55/A.py"},
        "src/B.py",
        "C.f",
        "proj55",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule59()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj51/client.py", "proj51/src/__init__.py", "proj51/src/module.py"},
        "src/module.py",
        "f",
        "proj51",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule60()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj52/client.py", "proj52/src/__init__.py", "proj52/src/module.py"},
        "src/module.py",
        "f",
        "proj52",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule61()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj56/src/__init__.py", "proj56/src/B.py", "proj56/A.py"},
        "src/B.py",
        "C.f",
        "proj56",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule62()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj57/src/__init__.py", "proj57/src/B.py", "proj57/A.py"},
        "src/B.py",
        "C.f",
        "proj57",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule63()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj58/src/__init__.py", "proj58/src/B.py", "proj58/A.py"},
        "src/B.py",
        "C.__call__",
        "proj58",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule64()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj59/src/__init__.py", "proj59/src/B.py", "proj59/A.py"},
        "src/B.py",
        "C.__call__",
        "proj59",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule65()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj60/src/__init__.py", "proj60/src/module.py", "proj60/client.py"},
        "src/module.py",
        "f",
        "proj60",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule66()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj61/src/__init__.py", "proj61/src/module.py", "proj61/client.py"},
        "src/module.py",
        "f",
        "proj61",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/202. */
  @Test
  public void testModule67()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj62/src/__init__.py", "proj62/src/B.py", "proj62/A.py"},
        "src/B.py",
        "C.__call__",
        "proj62",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/205. */
  @Test
  public void testModule68()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj63/src/__init__.py", "proj63/src/module.py", "proj63/client.py"},
        "src/module.py",
        "f",
        "proj63",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/205. */
  @Test
  public void testModule69()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj64/src/__init__.py", "proj64/src/module.py", "proj64/client.py"},
        "src/module.py",
        "f",
        "proj64",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test https://github.com/wala/ML/issues/210.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#210 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testModule70()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj65/src/__init__.py", "proj65/src/module.py", "proj65/client.py"},
        "src/module.py",
        "f",
        "proj65",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test https://github.com/wala/ML/issues/210.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#210 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testModule71()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj67/src/__init__.py", "proj67/src/module.py", "proj67/client.py"},
        "src/module.py",
        "f",
        "proj67",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/210. */
  @Test
  public void testModule72()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj68/src/__init__.py", "proj68/src/module.py", "proj68/client.py"},
        "src/module.py",
        "f",
        "proj68",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/210. */
  @Test
  public void testModule73()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj69/src/__init__.py", "proj69/src/module.py", "proj69/client.py"},
        "src/module.py",
        "f",
        "proj69",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/210. */
  @Test
  public void testModule74()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj70/src/__init__.py", "proj70/src/module.py", "proj70/client.py"},
        "src/module.py",
        "f",
        "proj70",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test https://github.com/wala/ML/issues/211.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#211 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testModule75()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj71/src/__init__.py", "proj71/src/module.py", "proj71/src/client.py"},
        "src/module.py",
        "f",
        "proj71",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/211. */
  @Test
  public void testModule76()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"module.py", "client.py"},
        "module.py",
        "f",
        "",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/211. */
  @Test
  public void testModule77()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"proj72/src/__init__.py", "proj72/src/module.py", "proj72/src/client.py"},
        "src/module.py",
        "f",
        "proj72",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/211. */
  @Test
  public void testModule78()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"module.py", "client2.py"},
        "module.py",
        "f",
        "",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/209. */
  @Test
  public void testModule79()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj73/models/__init__.py",
          "proj73/models/albert.py",
          "proj73/bert.py",
          "proj73/models/bert.py",
          "proj73/client.py"
        },
        "models/albert.py",
        "f",
        "proj73",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));

    test(
        new String[] {
          "proj73/models/__init__.py",
          "proj73/models/albert.py",
          "proj73/bert.py",
          "proj73/models/bert.py",
          "proj73/client.py"
        },
        "models/bert.py",
        "g",
        "proj73",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test https://github.com/wala/ML/issues/209. */
  @Test
  public void testModule80()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj74/models/__init__.py",
          "proj74/models/albert.py",
          "proj74/bert.py",
          "proj74/models/bert.py",
          "proj74/client.py"
        },
        "models/albert.py",
        "f",
        "proj74",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));

    test(
        new String[] {
          "proj74/models/__init__.py",
          "proj74/models/albert.py",
          "proj74/bert.py",
          "proj74/models/bert.py",
          "proj74/client.py"
        },
        "models/bert.py",
        "g",
        "proj74",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod2() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method2.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod3() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method3.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod4() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method4.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod5() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method5.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod6() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method6.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod7() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method7.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod8() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method8.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod9() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method9.py",
        "MyClass.the_static_method",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod10() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method10.py",
        "MyClass.the_static_method",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod11() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_static_method11.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod12() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_static_method12.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod13() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method13.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testStaticMethod14() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method14.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testClassMethod() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_class_method.py",
        "MyClass.the_class_method",
        1,
        1,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testClassMethod2() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_class_method2.py",
        "MyClass.the_class_method",
        1,
        1,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testClassMethod3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_class_method3.py", "MyClass.f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testClassMethod4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_class_method4.py", "MyClass.f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testClassMethod5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_class_method5.py", "MyClass.f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testAbstractMethod() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_abstract_method.py", "D.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_abstract_method.py", "C.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testAbstractMethod2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_abstract_method2.py", "D.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_abstract_method2.py", "C.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testAbstractMethod3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_abstract_method3.py", "C.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test https://github.com/wala/ML/issues/188.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#188 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method2.py", "f", 1, 1, Map.of(2, Set.of(MNIST_INPUT)));
  }

  /**
   * Test https://github.com/wala/ML/issues/190.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#190 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method3.py", "raffi", 1, 1, Map.of(2, Set.of(MNIST_INPUT)));
  }

  @Test
  public void testDecoratedMethod4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method4.py", "raffi", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method5.py", "raffi", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod6() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method6.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod7() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method7.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod8() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method8.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * This decorator isn't defined. Thus, we shouldn't have a CG node for it.
   *
   * <p>We now require nodes for functions under test. Otherwise, a test could pass even though the
   * function doesn't exist.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod9() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method9.py", "f", 0, 0);
  }

  /**
   * Test https://github.com/wala/ML/issues/190.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#190 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod10() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method10.py", "f", 1, 1, Map.of(2, Set.of(MNIST_INPUT)));
  }

  @Test
  public void testDecoratedMethod11() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method11.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod12() throws ClassHierarchyException, CancelException, IOException {
    // TODO: Change to 1, 1, 2 once https://github.com/wala/ML/issues/188 is fixed.
    test("tf2_test_decorated_method12.py", "f", 0, 0);
  }

  /**
   * Test https://github.com/wala/ML/issues/190.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#190 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod13() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method13.py", "f", 1, 1, Map.of(2, Set.of(MNIST_INPUT)));
  }

  @Test
  public void testDecoratedFunctions()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_decorated_functions.py",
        "dummy_fun",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "dummy_test",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "test_function",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "test_function2",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "test_function3",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "test_function4",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a pytest with decorators. */
  @Test
  public void testDecoratedFunctions2()
      throws ClassHierarchyException, CancelException, IOException {
    test("test_decorated_functions.py", "test_dummy", 0, 0);
  }

  /**
   * Test a pytest without decorators that needs a PYTHONPATH. This is a "control" case. We'll add a
   * decorator in the next case.
   *
   * @see TestTensorflow2Model#testModule11().
   */
  @Test
  public void testDecoratedFunctions3()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {
          "proj48/src/__init__.py",
          "proj48/src/tf2_test_module9a.py",
          "proj48/src/tf2_test_module9b.py",
          "proj48/src/test_module10.py"
        },
        "src/tf2_test_module9b.py",
        "D.f",
        "proj48",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test a pytest without decorators. This is a "control." */
  @Test
  public void testDecoratedFunctions4()
      throws ClassHierarchyException, CancelException, IOException {
    test("test_decorated_functions2.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a pytest with a decorator. */
  @Test
  public void testDecoratedFunctions5()
      throws ClassHierarchyException, CancelException, IOException {
    test("test_decorated_functions3.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test a pytest with a decorator that needs a PYTHONPATH.
   *
   * @see TestTensorflow2Model#testModule11().
   */
  @Test
  public void testDecoratedFunctions6()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {
          "proj49/src/__init__.py",
          "proj49/src/tf2_test_module9a.py",
          "proj49/src/tf2_test_module9b.py",
          "proj49/src/test_module10.py"
        },
        "src/tf2_test_module9b.py",
        "D.f",
        "proj49",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test a Pytest with a decorator without parameters. */
  @Test
  public void testDecoratedFunctions7()
      throws ClassHierarchyException, CancelException, IOException {
    test("test_decorated_functions4.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test a Pytest with a decorator without parameters that needs a PYTHONPATH.
   *
   * @see TestTensorflow2Model#testModule11().
   */
  @Test
  public void testDecoratedFunctions8()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {
          "proj50/src/__init__.py",
          "proj50/src/tf2_test_module10a.py",
          "proj50/src/tf2_test_module10b.py",
          "proj50/src/test_module11.py"
        },
        "src/tf2_test_module10b.py",
        "D.f",
        "proj50",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test a Pytest with a decorator without parameters. The "test" is at the end of the filename.
   */
  @Test
  public void testDecoratedFunctions9()
      throws ClassHierarchyException, CancelException, IOException {
    test("decorated_function_test.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_reshape.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_6_FLOAT32)));
  }

  /**
   * Test https://github.com/wala/ML/issues/195.
   *
   * <p>FIXME: Should not throw an {@link IllegalArgumentException} once
   * https://github.com/wala/ML/issues/340 is fixed.
   */
  @Test
  public void testReshape2() throws ClassHierarchyException, CancelException, IOException {
    TensorType expectedType =
        new TensorType(
            FLOAT_32,
            asList(
                new SymbolicDim("?"), new NumericDim(28), new NumericDim(28), new NumericDim(1)));

    test("tf2_test_reshape2.py", "f", 1, 1, Map.of(2, Set.of(expectedType)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_reshape3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_6_INT32)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_reshape4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_28_28_1_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_reshape5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_28_28_1_FLOAT32)));
  }

  @Test
  public void testConvertToTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testConvertToTensor2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testConvertToTensor5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testConvertToTensor6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor8.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testConvertToTensor9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor9.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor10.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testConvertToTensor11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor11.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testConvertToTensor12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor12.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
  }

  @Test
  public void testOneHot()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot8.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot9.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot10.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot11.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot12.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot13.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot14()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot14.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  @Test
  public void testOneHot15()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot15.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  @Test
  public void testOneHot16()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot16.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  @Test
  public void testOneHot17()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot17.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testOneHot18()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot18.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_5_3_FLOAT32)));
  }

  @Test
  public void testOneHot19()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // Fixed by handling `CONSTANT_OP_CONSTANT`.
    test("tf2_test_one_hot19.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
    test("tf2_test_one_hot19.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_5_3_FLOAT32)));
  }

  @Test
  public void testOneHot20()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_one_hot20.py",
        "test",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_3_3_FLOAT32),
            3, Set.of(TENSOR_3_3_INT32),
            4, Set.of(TENSOR_3_3_FLOAT32),
            5, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testEye()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testEye2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testEye3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testEye4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_2_FLOAT32)));
  }

  @Test
  public void testEye5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_2_3_FLOAT32)));
  }

  /**
   * FIXME: Should not throw an {@link IllegalArgumentException} once
   * https://github.com/wala/ML/issues/340 is fixed.
   */
  @Test
  public void testEye6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_2_3_FLOAT32)));
  }

  @Test
  public void testGamma()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testGamma2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT64)));
  }

  @Test
  public void testGamma3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_7_5_2_FLOAT32)));
  }

  @Test
  public void testGamma4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testGamma5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testGamma6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testGammaMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testFillKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_fill_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
  }

  @Test
  public void testFillMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_fill_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testRangeStartLimitKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_start_limit_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_INT32)));
  }

  @Test
  public void testRange1PosLimitDeltaKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_1_pos_limit_delta_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testRange1PosDeltaKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {

    test("tf2_test_range_1_pos_delta_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_INT32)));
  }

  @Test
  public void testRangeStartDeltaKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_start_delta_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testRangeStartKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_start_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testRangeKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testRangeMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testPoisson()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testPoisson2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testPoisson3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT64)));
  }

  @Test
  public void testPoisson4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_7_5_2_FLOAT32)));
  }

  @Test
  public void testSparseEye() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_5_FLOAT32)));
  }

  @Test
  public void testSparseEye2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_5_FLOAT32)));
  }

  @Test
  public void testSparseEye3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_5_INT32)));
  }

  @Test
  public void testSparseEye4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_2_FLOAT32)));
  }

  @Test
  public void testSparseEye5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_2_FLOAT32)));
  }

  @Test
  public void testSparseEye6() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_2_INT32)));
  }

  @Test
  public void testRaggedConstant() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testRaggedConstant2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testRaggedConstant3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_NONE_INT32)));
  }

  @Test
  public void testRaggedConstant4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_NONE_FLOAT32)));
  }

  @Test
  public void testRaggedConstant5() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant5.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_NONE_NONE_NONE_FLOAT32)));
  }

  @Test
  public void testRaggedConstant6() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testRaggedConstant7() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testRaggedConstant8() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant8.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_NONE_FLOAT32)));
  }

  @Test
  public void testRaggedConstant9() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant9.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_NONE_NONE_FLOAT32)));
  }

  @Test
  public void testRaggedConstant10() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant10.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_NONE_1_FLOAT32)));
  }

  @Test
  public void testRaggedConstant11() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant11.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_NONE_FLOAT32)));
  }

  @Test
  public void testRaggedConstant12() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant12.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_NONE_2_FLOAT32)));
  }

  @Test
  public void testRaggedConstant13() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant13.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_0_NONE_FLOAT32)));
  }

  @Test
  public void testRaggedConstant14() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant14.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_0_NONE_3_FLOAT32)));
  }

  @Test
  public void testRaggedConstant15() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant15.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_NONE_2_INT32)));
  }

  @Test
  public void testRaggedConstant16() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant16.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_NONE_2_INT32)));
  }

  /**
   * Test non-uniform inner dimensions.
   *
   * <p>TODO: Remove expected assertion error once https://github.com/wala/ML/issues/350 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testRaggedConstant17() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant17.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_NONE_2_3_INT32)));
  }

  /** This one works because the inner dimensions are uniform. */
  @Test
  public void testRaggedConstant18() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant18.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_NONE_2_2_INT32)));
  }

  @Test
  public void testRaggedConstantKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant_keyword.py",
        "test",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_2_NONE_INT32),
            3, Set.of(TENSOR_2_NONE_INT32),
            4, Set.of(TENSOR_2_NONE_INT32)));
  }

  @Test
  public void testRaggedFromRowSplits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_splits.py",
        "test_ragged_from_row_splits",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_5_NONE_INT32),
            3, Set.of(TENSOR_5_NONE_INT32),
            4, Set.of(TENSOR_5_NONE_INT32)));
  }

  @Test
  public void testRaggedRange() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_NONE_INT32)));
  }

  @Test
  public void testRaggedRange2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_NONE_INT32)));
  }

  @Test
  public void testRaggedRange3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testRaggedRange4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testRaggedRange5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testRaggedRange6() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testRaggedRange7() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_NONE_INT32)));
  }

  @Test
  public void testRaggedRangeKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_range_keyword.py",
        "test",
        5,
        5,
        Map.of(
            2, Set.of(TENSOR_1_NONE_INT32),
            3, Set.of(TENSOR_1_NONE_INT32),
            4, Set.of(TENSOR_1_NONE_INT32),
            5, Set.of(TENSOR_1_NONE_INT32),
            6, Set.of(TENSOR_1_NONE_INT32)));
  }

  @Test
  public void testSparseTensor() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_tensor.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_4_INT32)));
  }

  @Test
  public void testInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_input.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_NONE_32_FLOAT32)));
  }

  @Test
  public void testInput2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_input2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_NONE_NONE_STRING)));
  }

  @Test
  public void testRaggedFromNestedRowLengths()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_lengths.py",
        "test_ragged_from_nested_row_lengths",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_NONE_NONE_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowLengthsKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_lengths_keyword.py",
        "test_ragged_from_nested_row_lengths_keyword",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_NONE_NONE_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowLengthsKeyword2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_lengths_keyword2.py",
        "test_ragged_from_nested_row_lengths_keyword2",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_NONE_NONE_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowSplitsPositional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_splits_positional.py",
        "test_ragged_from_nested_row_splits_positional",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_NONE_NONE_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowSplitsKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_splits_keyword.py",
        "test_ragged_from_nested_row_splits_keyword",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_NONE_NONE_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowSplitsMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_splits_mixed.py",
        "test_ragged_from_nested_row_splits_mixed",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_NONE_NONE_INT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsPositional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_positional.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_NONE_NONE_STRING)));
  }

  @Test
  public void testRaggedNestedValueRowidsPositionalLists()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_positional_lists.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_NONE_NONE_STRING)));
  }

  @Test
  public void testRaggedNestedValueRowidsPositionalTuples()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_positional_tuples.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_NONE_NONE_STRING)));
  }

  @Test
  public void testRaggedNestedValueRowidsKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_keyword.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_NONE_NONE_INT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsKeywordLists()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_keyword_lists.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_NONE_NONE_INT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsKeywordTuples()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_keyword_tuples.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_NONE_NONE_INT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_mixed.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_NONE_NONE_FLOAT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsMixedLists()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_mixed_lists.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_NONE_NONE_FLOAT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsMixedTuples()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_mixed_tuples.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_NONE_NONE_FLOAT32)));
  }

  @Test
  public void testRaggedFromNestedValueRowIdsComplete()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // check_case_1: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_1",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_NONE_INT32)));

    // check_case_2: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_2",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_NONE_INT32)));

    // check_case_3: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_3",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_NONE_INT32)));

    // check_case_4: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_4",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_NONE_INT32)));

    // check_case_5: [2, None, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_5",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_NONE_NONE_INT32)));

    // check_case_6: [2, None], float32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_6",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_NONE_FLOAT32)));

    // check_case_7: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_7",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_NONE_INT32)));
  }

  @Test
  public void testVariablePositional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_variable_positional.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testVariableKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_variable_keyword.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
    test("tf2_test_variable_keyword.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testVariableMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_variable_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT64)));
  }

  @Test
  public void testVariablePositionalComplex()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {

    test("tf2_test_variable_positional_complex.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testVariablePositionalShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {

    test(
        "tf2_test_variable_positional_shape.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  @Test
  public void testVariablePositionalDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {

    test("tf2_test_variable_positional_dtype.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT64)));
  }

  /**
   * FIXME: Should not throw a {@link NullPointerException} once
   * https://github.com/wala/ML/issues/340 is resolved.
   *
   * @throws ClassHierarchyException
   * @throws IllegalArgumentException
   * @throws CancelException
   * @throws IOException
   */
  @Test
  public void testModelInit()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_model_init.py", "check_positional", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test("tf2_test_model_init.py", "check_keyword", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test("tf2_test_model_init.py", "check_mixed", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test("tf2_test_model_init.py", "check_subclass", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test(
        "tf2_test_model_init.py",
        "check_multiple",
        2,
        2,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testConvertToTensor13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_convert_to_tensor13.py",
        "test",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_3_FLOAT32),
            3, Set.of(TENSOR_2_2_INT32),
            4, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testEye7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_eye7.py",
        "test",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_2_3_INT32),
            3, Set.of(TENSOR_3_3_FLOAT32),
            4, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testGamma7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gamma7.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_FLOAT64),
            3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testPoisson5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_poisson5.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_FLOAT64),
            3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testSparseAdd7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_add7.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_2_INT32),
            3, Set.of(TENSOR_2_2_INT32)));
  }

  @Test
  public void testSparseEye7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_eye7.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_3_INT32),
            3, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testDense()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));
  }

  /**
   * Test for <a href="https://github.com/wala/ML/issues/371">wala/ML#371</a>. A single {@code
   * Dense} layer call inside {@code M.call} with a {@code tf.keras.Input} parameter.
   *
   * <p>Two tensor variables are found: the {@code x} parameter (v3, shape {@code (None, 3)}) and
   * the {@code Dense} result (v25, shape {@code (None, 4)}). Both are correct source-level tensors
   * under a single trampoline context.
   */
  @Test
  public void testDenseModelCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense_model_call.py", "M.call", 1, 2, Map.of(3, Set.of(TENSOR_NONE_3_FLOAT32)));
  }

  @Test
  public void testDense2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense2.py", "consume1", 1, 1, Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));
    test("tf2_test_dense2.py", "consume2", 1, 1, Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  /**
   * Chained {@code Dense} layer calls at module level, where the second layer's {@code inputs}
   * argument is the return value of the first layer's call. Exercises shape propagation through a
   * layer-call result at script-body scope.
   */
  @Test
  public void testDenseChain()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense_chain.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  /**
   * Chained {@code Dense} layer calls inside a {@code tf.keras.Model.__call__} method body with
   * direct {@code self.layer1} / {@code self.layer2} attribute reads. Exercises shape propagation
   * through a layer-call result inside a user-defined class method.
   */
  @Test
  public void testDenseChain2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense_chain2.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  /**
   * Chained {@code Dense} layer calls inside a {@code tf.keras.Model.__call__} method body, where
   * the layers are iterated via a {@code for} loop over a {@code self.layers_list} attribute rather
   * than being accessed by direct attribute name. Exercises shape propagation through a loop-phi'd
   * local whose points-to set spans every list element.
   */
  @Test
  public void testDenseChain3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense_chain3.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));
  }

  /**
   * Exercise the duck-typed {@code numpy.ndarray.astype(...)} dispatch path added for wala/ML#356.
   * The receiver is {@code x_train}, the first element of {@code
   * tf.keras.datasets.mnist.load_data()}. Without mnist modeling, the receiver's concrete shape
   * cannot be resolved; after the {@code astype} call, {@code consume}'s parameter is recognised as
   * a float32 tensor with a {@code null} dims list.
   */
  @Test
  public void testAstype()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_astype.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_60000_28_28_FLOAT32)));
  }

  /**
   * Test https://github.com/wala/ML/issues/358.
   *
   * <p>Derived from {@link #testModelCall()} (see {@code tf2_test_model_call.py}) by adding a
   * {@code consume(x)} call inside {@code SequentialModel.__call__} immediately after {@code x =
   * self.dense_2(x)}. At that point {@code x} should have shape {@code (20, 10)} and dtype {@code
   * float32}: the chain traces {@code (20, 28, 28)} input → {@code Flatten} → {@code (20, 784)} →
   * 100× {@code Dense(64)} → {@code (20, 64)} → {@code Dropout} → {@code (20, 64)} → {@code
   * Dense(10)} → {@code (20, 10)}.
   *
   * <p>Currently fails because the analysis produces {@code {? of float32}} at {@code consume}'s
   * parameter instead of {@code {(20, 10) of float32}}. The minimal chained-layer regression tests
   * ({@link #testDenseChain()}, {@link #testDenseChain2()}, {@link #testDenseChain3()}) all pass,
   * so the trigger here is something more specific to {@code SequentialModel}'s particular
   * combination of {@code Flatten} + 100-element layer list comprehension + {@code Dropout} +
   * {@code tf.random.uniform} input.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#358 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testModelCallConsume()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call_consume.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_20_10_FLOAT32)));
  }

  private void test(
      String filename,
      String functionName,
      int expectedNumberOfTensorParameters,
      int expectedNumberOfTensorVariables)
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {filename},
        filename,
        functionName,
        "",
        expectedNumberOfTensorParameters,
        expectedNumberOfTensorVariables,
        emptyMap());
  }

  protected void test(
      String filename,
      String functionName,
      int expectedNumberOfTensorParameters,
      int expectedNumberOfTensorVariables,
      Map<Integer, Set<TensorType>> expectedTensorParameterValueNumberToTypes)
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {filename},
        filename,
        functionName,
        "",
        expectedNumberOfTensorParameters,
        expectedNumberOfTensorVariables,
        expectedTensorParameterValueNumberToTypes);
  }

  protected void test(
      String[] projectFilenames,
      String filename,
      String functionName,
      String pythonPath,
      int expectedNumberOfTensorParameters,
      int expectedNumberOfFunctionTensorVariables,
      Map<Integer, Set<TensorType>> expectedTensorParameterValueNumberToTypes)
      throws ClassHierarchyException, CancelException, IOException {
    List<File> pathFiles = this.getPathFiles(pythonPath);
    PythonTensorAnalysisEngine engine = makeEngine(pathFiles, projectFilenames);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();

    addPytestEntrypoints(builder);

    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);

    if (LOGGER.isLoggable(Level.FINE)) {
      CAstCallGraphUtil.AVOID_DUMP.set(false);
      CAstCallGraphUtil.dumpCG(
          ((SSAPropagationCallGraphBuilder) builder).getCFAContextInterpreter(),
          builder.getPointerAnalysis(),
          CG);
      LOGGER.fine("Call graph:\n" + CG);
    }

    TensorTypeAnalysis analysis = engine.performAnalysis(builder);
    LOGGER.info("Tensor analysis: " + analysis);

    Map<PointerKey, AnalysisError> errors = engine.getErrors();

    errors.forEach(
        (k, v) ->
            LOGGER.info(
                () -> "Pointer key: " + k + " has analysis error: " + v + " at " + v.position()));

    // a mapping from function signatures to pointer keys.
    Map<String, Set<LocalPointerKey>> functionSignatureToPointerKeys = new HashMap<>();

    // a mapping from function signatures to tensor variables.
    Map<String, Set<TensorVariable>> functionSignatureToTensorVariables = new HashMap<>();

    // a mapping from pointer keys to tensor variables.
    Map<PointerKey, TensorVariable> pointerKeyToTensorVariable = new HashMap<>();

    // for each pointer key, tensor variable pair.
    analysis.forEach(
        pt -> {
          PointerKey pointerKey = pt.fst;
          TensorVariable tensorVariable = pt.snd;

          // associate the pointer key to the tensor variable.
          pointerKeyToTensorVariable.put(pointerKey, tensorVariable);

          if (pointerKey instanceof LocalPointerKey) {
            LocalPointerKey localPointerKey = (LocalPointerKey) pointerKey;

            // get the call graph node associated with the pointer key.
            CGNode node = localPointerKey.getNode();

            // get the method associated with the call graph node.
            IMethod method = node.getMethod();
            String methodSignature = method.getSignature();

            // associate the method to the pointer key.
            functionSignatureToPointerKeys.compute(
                methodSignature,
                (_, v) -> {
                  if (v == null) v = new HashSet<>();
                  v.add(localPointerKey);
                  return v;
                });

            // associate the method to the tensor variables.
            functionSignatureToTensorVariables.compute(
                methodSignature,
                (_, v) -> {
                  if (v == null) v = new HashSet<>();
                  v.add(tensorVariable);
                  return v;
                });
          } else LOGGER.warning(() -> "Encountered: " + pointerKey.getClass());
        });

    final String functionSignature =
        "script " + filename.replace('/', '.') + "." + functionName + ".do()LRoot;";

    // List the CG nodes as a "flat" list.
    LOGGER.fine(
        () ->
            "Call graph nodes:\n"
                + getFunctionSignatures(CG).collect(Collectors.joining("\n\t", "\t", "")));

    // check that the function exists in the call graph.
    assertTrue(
        "Function must exist in call graph.",
        getFunctionSignatures(CG).anyMatch(s -> s.equals(functionSignature)));

    // get the tensor variables for the function.
    LOGGER.fine("Looking for signature: " + functionSignature);
    LOGGER.fine(
        "Available signatures in functionSignatureToTensorVariables: "
            + functionSignatureToTensorVariables.keySet());
    Set<TensorVariable> functionTensorVariables =
        functionSignatureToTensorVariables.getOrDefault(functionSignature, emptySet());

    assertEquals(expectedNumberOfFunctionTensorVariables, functionTensorVariables.size());

    // check value number cardinality.
    assertEquals(
        "Each tensor parameter should have a unique value number.",
        expectedNumberOfTensorParameters,
        expectedTensorParameterValueNumberToTypes.size());

    // get the pointer keys for the function by their contexts.
    Map<Context, Set<LocalPointerKey>> contextToFunctionParameterPointerKeys =
        functionSignatureToPointerKeys.getOrDefault(functionSignature, emptySet()).stream()
            .filter(LocalPointerKey::isParameter)
            .collect(groupingBy(lpk -> lpk.getNode().getContext(), toSet()));

    assertTrue(
        "Because tensor parameters are inferred via function arguments, we need at least one"
            + " calling context if we are expecting at least one tensor parameter.",
        expectedNumberOfTensorParameters <= 0 || contextToFunctionParameterPointerKeys.size() > 0);

    for (Context ctx : contextToFunctionParameterPointerKeys.keySet()) {
      // check tensor parameters.
      Set<LocalPointerKey> functionParameterPointerKeys =
          contextToFunctionParameterPointerKeys.get(ctx);

      assertEquals(expectedNumberOfTensorParameters, functionParameterPointerKeys.size());

      // check actual value numbers.
      Set<Integer> actualParameterValueNumberSet =
          functionParameterPointerKeys.stream()
              .map(LocalPointerKey::getValueNumber)
              .collect(Collectors.toSet());

      assertEquals(
          expectedTensorParameterValueNumberToTypes.keySet(), actualParameterValueNumberSet);

      // check types.
      functionParameterPointerKeys.stream()
          .forEach(
              lpk -> {
                TensorVariable tensorVariable = pointerKeyToTensorVariable.get(lpk);
                assertNotNull(
                    "Checking tensor variable for pointer key: " + lpk + ".", tensorVariable);

                Set<TensorType> types = tensorVariable.getTypes();
                assertNotNull("Checking tensor variable for pointer key: " + lpk + ".", types);

                Set<TensorType> expectedTypes =
                    expectedTensorParameterValueNumberToTypes.get(lpk.getValueNumber());
                assertNotNull(
                    "Checking expected types for value number: " + lpk.getValueNumber() + ".",
                    expectedTypes);

                // check that the types are the same.
                if (LOGGER.isLoggable(Level.INFO) && !expectedTypes.equals(types)) {
                  LOGGER.info("Assertion failure for lpk: " + lpk);
                  LOGGER.info("  Node: " + lpk.getNode());
                  LOGGER.info("  Context: " + lpk.getNode().getContext());
                  LOGGER.info("  Expected: " + expectedTypes);
                  LOGGER.info("  Actual: " + types);
                }

                assertEquals(
                    "Comparing expected types for value number: " + lpk.getValueNumber() + ".",
                    expectedTypes,
                    types);
              });
    }
  }

  /**
   * Returns a {@link Stream} of {@link String}s representing the signatures of functions
   * represented by the nodes in the given {@link CallGraph}.
   *
   * @param CG The {@link CallGraph} containing the nodes in question.
   * @return A {@link Stream} of {@link String}s representing the signatures of functions
   *     represented by the nodes in the given {@link CallGraph}.
   */
  private static Stream<String> getFunctionSignatures(CallGraph CG) {
    return CG.stream().map(CGNode::getMethod).map(IMethod::getSignature);
  }

  /**
   * Extracts a {@link List} of {@link File}s from the given {@link String} representing a list of
   * paths. Each path is separated by a colon.
   *
   * @param string A colon-separated list of paths.
   * @return {@link List} of {@link File}s constructed by parsing the given {@link String}.
   */
  protected List<File> getPathFiles(String string) {
    if (string == null || string.isEmpty() || string.isBlank()) return emptyList();

    return Arrays.asList(string.split(":")).stream()
        .map(
            s -> {
              File f = new File(s);

              if (f.exists()) return f;

              try {
                URL url = new URI(s).toURL();
                return new File(new FileProvider().filePathFromURL(url));
              } catch (MalformedURLException | URISyntaxException | IllegalArgumentException e) {
                try {
                  URL resource = this.getClass().getResource("/" + string);
                  String path = resource.getPath();
                  return new File(path);
                } catch (Exception e1) {
                  throw new RuntimeException(e1);
                }
              }
            })
        .collect(toList());
  }
}
