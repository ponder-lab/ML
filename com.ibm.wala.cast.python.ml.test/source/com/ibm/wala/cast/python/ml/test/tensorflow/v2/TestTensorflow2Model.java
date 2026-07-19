package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_STRING;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SPARSE_TENSOR_4_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_1_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_28_28_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_32_32_3_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_100_784_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_102_13_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_102_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10_2_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_16_28_28_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_0_0_9_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_10_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_2_27_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2246_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2246_OBJECT;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_25000_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_25000_OBJECT;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_256_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_28_28_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_28_28_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_64_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_256_784_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_4_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_4_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_6_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_30_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_32_28_28_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_5_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_STRING;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_404_13_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_404_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4096_32_32_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4096_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_10_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_1_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_8_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_50000_1_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_50000_32_32_3_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_28_28_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_28_28_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_784_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_6_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_6_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_7_5_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_8982_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_8982_OBJECT;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_96_28_28_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_DYNAMIC_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_INT32_UNKNOWN_SHAPE;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.UNKNOWN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.COMPLEX64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static java.util.Collections.emptyMap;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.BroadcastTo;
import com.ibm.wala.cast.python.ml.client.Linspace;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.SparseTensorType;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConcreteTypeKey;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.intset.OrdinalSet;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/** Test TF2 APIs. */
public class TestTensorflow2Model extends AbstractTensorTest {

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

  /**
   * The {@code returned(a)} parameter is {@code a = tf.constant([1, 1.0])}, i.e. {@code (2,)
   * float32}. The asserted set is a union across contexts (per the test helper's union-per-vn
   * semantics): the {@code (2,) float32} is the parameter's real type, now precise after the top_k
   * output-shape composer (<a href="https://github.com/wala/ML/issues/609">wala/ML#609</a>)
   * sharpened the previous ⊤. The {@code (2,) int32} is a top_k {@code indices} output leaking in
   * through a collapsed 1-CFA context (it was already present in the old union as ⊤ int32); the
   * composer only made it concrete. The leak itself is a context-sensitivity artifact.
   */
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

  /**
   * Regression test for wala/ML#452: iterating a {@code tf.data.TextLineDataset} via {@code for
   * element in dataset:} should classify each element as a 0-D string tensor. The receiving
   * function {@code func}'s parameter at {@code vn=2} must therefore have type {@code
   * SCALAR_TENSOR_OF_STRING}. Pre-fix, the analyzer's {@code TextLineDataset} model didn't preserve
   * the per-element tensor type through the iteration substrate, leaving {@code func} with no
   * tensor classification at all (downstream {@code Function.getHasTensorParameter()} reported
   * false).
   */
  @Test
  public void testTextLineDatasetIterationElementType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_textlinedataset_iter.py",
        "func",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   *
   * <p>{@code summarize_weights(step)} takes only an {@code int} step counter (not a tensor, hence
   * parameter count 0). Inside the function, dict lookups {@code weights[w]} and {@code biases[b]}
   * feed 4 tensor variables to {@code tf.summary.histogram}. Previously this test registered 5: the
   * extra was a spurious {@code v2} (the {@code step} parameter) typed {@code {(256, 784) float32}}
   * via PA-graph propagation from the {@code enumerate()} call at the caller ({@code for step, (x,
   * y) in enumerate(ds, 1):}). The leak was at the PA substrate &mdash; tensor types flowing to the
   * enumerate tuple's field-0 slot through the PTS-graph edge, not at generator dispatch.
   *
   * <p>Closed by wala/ML#409: {@link com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine}
   * now detects enumerate-first-field reads structurally and pins their {@link
   * com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis} state to empty via a {@code DropOp}
   * edge transfer that both clears existing leaked state and FIXes the slot against further
   * predecessor updates. Count correctly reports 4.
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

  /**
   * {@code add(a, b)} is called only as {@code add(list, list)} where {@code list = [tf.ones([1,
   * 2]), tf.ones([2, 2])]}, so {@code a} and {@code b} are Python lists. {@code a + b} is therefore
   * list concatenation, not a tensor add, so {@code add} has no tensor variables. Before
   * wala/ML#750 the {@code list} operands' allocations were counted as tensor evidence and {@code a
   * + b} was falsely typed as a tensor (the local count was 1); the concatenation is now correctly
   * not a tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testTensorList2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensor_list2.py", "add", 0, 0);
  }

  @Test
  public void testTensorList3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // `list.append` is modeled (wala/ML#136): both appended tensors reach `add` through the
    // iteration, so each parameter carries the union of the two shapes, as in testTensorList.
    test(
        "tf2_test_tensor_list3.py",
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
  public void testTensorList4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensor_list4.py", "add", 0, 0);
  }

  @Test
  public void testTensorList5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensor_list5.py", "add", 0, 0);
  }

  /**
   * Regression guard for wala/ML#655: a {@code tf.io.FixedLenFeature} value, parsed by {@code
   * tf.io.parse_single_example} and read back through a dict subscript, types as a dense tensor
   * whose shape comes from the feature's {@code dims} argument and whose dtype comes from its
   * {@code type} argument. Previously {@code FixedLenFeature.do} allocated an {@code
   * Ltensorflow/objects/ feature} that the manual tensor walker ignores, so the parsed value never
   * typed.
   */
  @Test
  public void testFixedLenFeature()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_fixed_len_feature.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 128))));
  }

  /**
   * Regression guard for wala/ML#655, end to end: the NLPGNN {@code TFLoader.load_valid} input
   * pipeline — a {@code TFRecordDataset} mapped by a {@code parse_single_example} decoder returning
   * a 4-tuple of {@code FixedLenFeature} dict-subscripts, then {@code prefetch}ed, then iterated
   * with a 4-way tuple unpack — types its first parsed field {@code X} ({@code input_ids}) to
   * {@code (128,)} int64. Before the {@code FixedLenFeature} fix, the mapped element was
   * non-tensor, so {@code X} (and any model parameter fed from it) came back non-tensor.
   */
  @Test
  public void testTfrecordLoader()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tfrecord_loader.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 128))));
  }

  /**
   * Regression guard for wala/ML#655, the full symptom-A chain in one fixture: the NLPGNN {@code
   * BilstmAttention} model (whose {@code predict} forwards to {@code self(inputs, training)} and
   * whose {@code call} delegates to a user-defined child {@code BiLSTM} layer built from unmodeled
   * sublayers) fed from the {@code TFLoader} {@code FixedLenFeature}/{@code TFRecordDataset}
   * source. The child {@code BiLSTM.call}'s {@code inputs} parameter types to {@code (128,)} int64,
   * flowing the parsed {@code input_ids} field through {@code model.predict(X)} → {@code __call__}
   * → {@code call} → {@code self.bilstm(inputs, training)}. This demonstrates that the symptom-A
   * cases are unit-reproducible via the source pipeline (not the affected file, which carries no
   * call site or source) and that the cause was the source typing, not the {@code __call__}
   * forwarding the issue title hypothesized. Before the {@code FixedLenFeature} fix, {@code inputs}
   * came back non-tensor.
   */
  @Test
  public void testBilstmLoaderE2e()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_bilstm_loader_e2e.py",
        "BiLSTM.call",
        1,
        2,
        Map.of(3, Set.of(TensorType.of(INT_64, 128))));
  }

  /**
   * Pins the <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a> {@code BiLSTM.call}
   * row on the vendored subject: the {@code BiLSTM} layer is vendored verbatim from {@code
   * kyzhouhzau/NLPGNN} ({@code nlpgnn/layers/bilstm.py}); only the driver is bespoke. The {@code
   * inputs} parameter (an integer token-ID tensor feeding a Keras {@code Embedding}) recovers
   * {@code (2, 5)} int32 exactly, flowing from the driver's {@code layer(tokens, training=False)}
   * call site through {@code tf.keras.layers.Layer.__call__} dispatch.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testBilstmCallVendored()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "bilstm_proj/nlpgnn/__init__.py",
          "bilstm_proj/nlpgnn/layers/__init__.py",
          "bilstm_proj/nlpgnn/layers/bilstm.py",
          "bilstm_proj/tf2_test_bilstm_call.py"
        },
        "nlpgnn/layers/bilstm.py",
        "BiLSTM.call",
        "bilstm_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Pins the <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a> {@code
   * TextCNN.predict} row on the vendored subject ({@code kyzhouhzau/NLPGNN}, {@code
   * nlpgnn/models/TextCNN.py}, same vendoring as {@link #testTextcnnCall()}): {@code predict}
   * forwards {@code inputs} to the model through {@code self(inputs, training)}. The {@code inputs}
   * parameter recovers {@code (2, 5)} int32 exactly; the second tracked local is the forward
   * result, float32 with ⊤ shape (the chained-layer body, wala/ML#358/wala/ML#530).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTextcnnPredict()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "textcnn_proj/nlpgnn/__init__.py",
          "textcnn_proj/nlpgnn/models/__init__.py",
          "textcnn_proj/nlpgnn/models/TextCNN.py",
          "textcnn_proj/tf2_test_textcnn_predict.py"
        },
        "nlpgnn/models/TextCNN.py",
        "TextCNN.predict",
        "textcnn_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Probe for <a href="https://github.com/wala/ML/issues/688">wala/ML#688</a>: a {@code map} stage
   * returning a tuple, batched with a tuple {@code padded_shapes}, iterated with destructuring —
   * the vendored gpt-2 {@code input_fn} element shape in miniature. Both halves type the
   * runtime-true {@code (4, 3) int64} (batch 4, padded to the longest sequence): the computed
   * second member resolves through the wala/ML#688 SSA fallback and the batch stage applies.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPaddedBatchPair()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_padded_batch_pair.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(DType.INT64, 4, 3))));
  }

  /**
   * Sibling half of {@link #testPaddedBatchPair()} (wala/ML#688): the second tuple member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPaddedBatchPair2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_padded_batch_pair.py",
        "consume2",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(DType.INT64, 4, 3))));
  }

  /**
   * Pins the <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a> {@code
   * TuckERLoader.target_convert} row on the vendored subject: the loader is vendored verbatim from
   * {@code kyzhouhzau/NLPGNN} ({@code nlpgnn/datas/graphloader.py}); the driver, the tiny {@code
   * data/} triple files, and the {@code nlpgnn/gnn/utils.py} reachable slice are bespoke. The
   * {@code targets} parameter (a {@code padded_batch} dict-element field) types {@code (2, ?)}
   * int32 — the declared {@code padded_shapes} dims under the batch dimension (<a
   * href="https://github.com/wala/ML/issues/673">wala/ML#673</a>) — unioned with the standard
   * partial-batch sibling {@code (?, ?)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   *     <p>The local count excludes the {@code enumerate(targets)} iterator object, pinned
   *     non-tensor since wala/ML#732; the element and its TensorFlow uses stay typed.
   */
  @Test
  public void testTuckerTargetConvert()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "tucker_proj/nlpgnn/__init__.py",
          "tucker_proj/nlpgnn/gnn/__init__.py",
          "tucker_proj/nlpgnn/gnn/utils.py",
          "tucker_proj/nlpgnn/datas/__init__.py",
          "tucker_proj/nlpgnn/datas/graphloader.py",
          "tucker_proj/tf2_test_tucker_target_convert.py"
        },
        "nlpgnn/datas/graphloader.py",
        "TuckERLoader.target_convert",
        "tucker_proj",
        1,
        6,
        Map.of(
            3,
            Set.of(
                new TensorType(INT_32, asList(new NumericDim(2), DynamicDim.INSTANCE)),
                new TensorType(INT_32, asList(new SymbolicDim("?"), DynamicDim.INSTANCE)))));
  }

  /**
   * Whole-project-layout probe for <a href="https://github.com/wala/ML/issues/678">wala/ML#678</a>:
   * the subject's structure in miniature — nested entry scripts ({@code tests/TG/EN/}) each
   * defining a same-named {@code GenGPT2} over a root-level {@code nlpgnn} package (inner {@code
   * gpt2.GPT2} model, shared closure-dispatching {@code samples.sample_sequence}) — the layout the
   * fixture-scale reproductions lacked. Both siblings keep their call-graph nodes and type — the
   * layout is excluded as the wala/ML#678 trigger — with the wala/ML#685 cross-sibling closure
   * union as the pinned imprecision.
   *
   * <p>TODO: Expect each sibling's own shape once <a
   * href="https://github.com/wala/ML/issues/685">wala/ML#685</a> keys closure callees on the
   * dispatched function object.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnSliceGeneration()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "nlpgnn_slice_proj/nlpgnn/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/callbacks.py",
          "nlpgnn_slice_proj/nlpgnn/models/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/models/gpt2.py",
          "nlpgnn_slice_proj/nlpgnn/sample/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/sample/samples.py",
          "nlpgnn_slice_proj/tests/__init__.py",
          "nlpgnn_slice_proj/tests/TG/__init__.py",
          "nlpgnn_slice_proj/tests/TG/EN/__init__.py",
          "nlpgnn_slice_proj/tests/TG/EN/generation.py",
          "nlpgnn_slice_proj/tests/TG/EN/interactive.py"
        },
        "tests/TG/EN/generation.py",
        "GenGPT2.predict",
        "nlpgnn_slice_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_2_FLOAT32, TENSOR_3_3_FLOAT32)));
  }

  /**
   * Sibling half of {@link #testNlpgnnSliceGeneration()} (wala/ML#678) — the {@code interactive}
   * entry script, the one degraded in the whole-project consumer runs.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnSliceInteractive()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "nlpgnn_slice_proj/nlpgnn/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/callbacks.py",
          "nlpgnn_slice_proj/nlpgnn/models/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/models/gpt2.py",
          "nlpgnn_slice_proj/nlpgnn/sample/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/sample/samples.py",
          "nlpgnn_slice_proj/tests/__init__.py",
          "nlpgnn_slice_proj/tests/TG/__init__.py",
          "nlpgnn_slice_proj/tests/TG/EN/__init__.py",
          "nlpgnn_slice_proj/tests/TG/EN/generation.py",
          "nlpgnn_slice_proj/tests/TG/EN/interactive.py"
        },
        "tests/TG/EN/interactive.py",
        "GenGPT2.predict",
        "nlpgnn_slice_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_2_FLOAT32, TENSOR_3_3_FLOAT32)));
  }

  /**
   * The complete NLPGNN subject vendored verbatim under {@code nlpgnn_full_proj} (wala/ML#690);
   * shared by {@link #testNlpgnnFullGeneration()} and {@link #testNlpgnnFullInteractive()} so the
   * two sibling guards cannot diverge as the fixture changes.
   */
  private static final String[] NLPGNN_FULL_PROJECT_FILES = {
    "nlpgnn_full_proj/nlpgnn/__init__.py",
    "nlpgnn_full_proj/nlpgnn/abandoned/GCNConvv0.py",
    "nlpgnn_full_proj/nlpgnn/abandoned/__init__.py",
    "nlpgnn_full_proj/nlpgnn/abandoned/scatter.py",
    "nlpgnn_full_proj/nlpgnn/bpemd/bpe.py",
    "nlpgnn_full_proj/nlpgnn/callbacks.py",
    "nlpgnn_full_proj/nlpgnn/datas/__init__.py",
    "nlpgnn_full_proj/nlpgnn/datas/checkpoint.py",
    "nlpgnn_full_proj/nlpgnn/datas/dataloader.py",
    "nlpgnn_full_proj/nlpgnn/datas/graphloader.py",
    "nlpgnn_full_proj/nlpgnn/datas/word2vec.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GAAEConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GATConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GCNConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GINConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GSConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/RGCNConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/TGCNConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/__init__.py",
    "nlpgnn_full_proj/nlpgnn/gnn/glob.py",
    "nlpgnn_full_proj/nlpgnn/gnn/messagepassing.py",
    "nlpgnn_full_proj/nlpgnn/gnn/utils.py",
    "nlpgnn_full_proj/nlpgnn/layers/__init__.py",
    "nlpgnn_full_proj/nlpgnn/layers/albert_transformer.py",
    "nlpgnn_full_proj/nlpgnn/layers/attention.py",
    "nlpgnn_full_proj/nlpgnn/layers/bilstm.py",
    "nlpgnn_full_proj/nlpgnn/layers/decoder.py",
    "nlpgnn_full_proj/nlpgnn/layers/dense.py",
    "nlpgnn_full_proj/nlpgnn/layers/embedding.py",
    "nlpgnn_full_proj/nlpgnn/layers/gpt2_transformer.py",
    "nlpgnn_full_proj/nlpgnn/layers/normalization.py",
    "nlpgnn_full_proj/nlpgnn/layers/transformer.py",
    "nlpgnn_full_proj/nlpgnn/metrics/Losess.py",
    "nlpgnn_full_proj/nlpgnn/metrics/Metric.py",
    "nlpgnn_full_proj/nlpgnn/metrics/__init__.py",
    "nlpgnn_full_proj/nlpgnn/metrics/crf.py",
    "nlpgnn_full_proj/nlpgnn/metrics/type.py",
    "nlpgnn_full_proj/nlpgnn/models/GAAE.py",
    "nlpgnn_full_proj/nlpgnn/models/GAT.py",
    "nlpgnn_full_proj/nlpgnn/models/GCN.py",
    "nlpgnn_full_proj/nlpgnn/models/GIN.py",
    "nlpgnn_full_proj/nlpgnn/models/GraphSage.py",
    "nlpgnn_full_proj/nlpgnn/models/PCNN.py",
    "nlpgnn_full_proj/nlpgnn/models/RGCN.py",
    "nlpgnn_full_proj/nlpgnn/models/TextCNN.py",
    "nlpgnn_full_proj/nlpgnn/models/TextGCN2019.py",
    "nlpgnn_full_proj/nlpgnn/models/__init__.py",
    "nlpgnn_full_proj/nlpgnn/models/albert.py",
    "nlpgnn_full_proj/nlpgnn/models/bert.py",
    "nlpgnn_full_proj/nlpgnn/models/gpt2.py",
    "nlpgnn_full_proj/nlpgnn/models/tucker.py",
    "nlpgnn_full_proj/nlpgnn/optimizers/__init__.py",
    "nlpgnn_full_proj/nlpgnn/optimizers/optim.py",
    "nlpgnn_full_proj/nlpgnn/sample/__init__.py",
    "nlpgnn_full_proj/nlpgnn/sample/samples.py",
    "nlpgnn_full_proj/nlpgnn/savers.py",
    "nlpgnn_full_proj/nlpgnn/tokenizers/__init__.py",
    "nlpgnn_full_proj/nlpgnn/tokenizers/gpt2_tokenization.py",
    "nlpgnn_full_proj/nlpgnn/tokenizers/tokenization.py",
    "nlpgnn_full_proj/nlpgnn/tools.py",
    "nlpgnn_full_proj/setup.py",
    "nlpgnn_full_proj/tests/CLS/ALBERT/albert_cls_test.py",
    "nlpgnn_full_proj/tests/CLS/ALBERT/albert_cls_train.py",
    "nlpgnn_full_proj/tests/CLS/BERT/bert_classification_test.py",
    "nlpgnn_full_proj/tests/CLS/BERT/bert_classification_train.py",
    "nlpgnn_full_proj/tests/CLS/BilstmAttention/bilstm_attention_test.py",
    "nlpgnn_full_proj/tests/CLS/BilstmAttention/bilstm_attention_train.py",
    "nlpgnn_full_proj/tests/CLS/TextCNN/text_cnn_test.py",
    "nlpgnn_full_proj/tests/CLS/TextCNN/text_cnn_train.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/attention.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/bert.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/build_graph.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/train_text_gcn.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/transformer.py",
    "nlpgnn_full_proj/tests/GNN/auto_encoder/GAAE.py",
    "nlpgnn_full_proj/tests/GNN/gnn_for_nlp/text_gcn.py",
    "nlpgnn_full_proj/tests/GNN/gnn_for_nlp/text_sage.py",
    "nlpgnn_full_proj/tests/GNN/nodes_graph_classfication/train_gan.py",
    "nlpgnn_full_proj/tests/GNN/nodes_graph_classfication/train_gcn.py",
    "nlpgnn_full_proj/tests/GNN/nodes_graph_classfication/train_gin.py",
    "nlpgnn_full_proj/tests/GNN/nodes_graph_classfication/train_graphsage.py",
    "nlpgnn_full_proj/tests/KG2E/run_tucker.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/albert_ner_test.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/albert_ner_train.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/bert_ner_test.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/bert_ner_train.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/data_processing.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/bert_ner_crf_test.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/bert_ner_crf_train.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/bert_ner_test.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/bert_ner_train.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/ner_data_preprocess.py",
    "nlpgnn_full_proj/tests/TG/EN/generation.py",
    "nlpgnn_full_proj/tests/TG/EN/interactive.py"
  };

  /**
   * The complete MusicTransformer-tensorflow2.0 subject vendored verbatim under {@code
   * musictransformer_proj} (wala/ML#683, wala/ML#684), so its whole-project guards analyze the
   * subject shape rather than a distilled fixture.
   */
  private static final String[] MUSICTRANSFORMER_PROJECT_FILES = {
    "musictransformer_proj/custom/callback.py",
    "musictransformer_proj/custom/layers.py",
    "musictransformer_proj/data.py",
    "musictransformer_proj/deprecated/seq_test.py",
    "musictransformer_proj/deprecated/sequence.py",
    "musictransformer_proj/deprecated/train.py",
    "musictransformer_proj/dist_train.py",
    "musictransformer_proj/generate.py",
    "musictransformer_proj/model.py",
    "musictransformer_proj/params.py",
    "musictransformer_proj/preprocess.py",
    "musictransformer_proj/train.py",
    "musictransformer_proj/utils.py"
  };

  /**
   * Verbatim whole-project guard for <a
   * href="https://github.com/wala/ML/issues/690">wala/ML#690</a>: the full NLPGNN subject vendored
   * as-is (all 94 {@code .py} files, matching the consumer's whole-project run; no added {@code
   * __init__.py}s under {@code tests/}). Before the fix, whichever same-named {@code GenGPT2}
   * sibling's closure reached the shared {@code sample_sequence.step} node second lost its lexical
   * {@code model} wiring (WALA's one-shot {@code visitLexical} snapshot), so its {@code
   * predict}/{@code call} never dispatched and its method nodes vanished at whole-project scale.
   * This pins the generation sibling's {@code predict} node and its parameter type, which must be
   * symmetric with {@link #testNlpgnnFullInteractive()}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullGeneration()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    runNlpgnnFullGeneration();
  }

  /**
   * Runs the NLPGNN whole-project generation analysis with its call-graph and type assertions.
   * Package-visible so {@link DiagnosticLoggingVolumeTest} can rerun the analysis under {@code
   * FINEST} without invoking another class's {@code @Test} method.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  public void runNlpgnnFullGeneration()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "tests/TG/EN/generation.py",
        "GenGPT2.predict",
        "nlpgnn_full_proj",
        1,
        1,
        Map.of(
            3, Set.of(new TensorType(UNKNOWN, asList(UnresolvedDim.INSTANCE, new NumericDim(1))))));
  }

  /**
   * In-vivo anchor for wala/ML#704: the vendored NLPGNN {@code einsum_via_matmul} ({@code
   * nlpgnn/layers/dense.py}). The {@code input_tensor} parameter now carries concrete batch and
   * sequence dimensions with a dynamic trailing (hidden) dimension, delivered from the entry
   * scripts' explicit {@code model.build} contracts through the embedding's output reshape
   * (wala/ML#716, wala/ML#717): {@code (8, 100)} and {@code (8, 10)} leading pairs from the entries
   * whose pipelines reach this layer, each with the trailing {@code input_shape[-1] *
   * self.embedding_size} element unresolved, since the factor comes from a checkpoint config the
   * analysis cannot read — a fixed runtime size of unknown value ({@link UnresolvedDim},
   * wala/ML#721), not a runtime-{@code None} axis. The rank-2 {@code (8, D)} member is the
   * embedding guard-φ's path-insensitive phantom (the pre-{@code expand_dims} member). The {@code
   * tf.reshape}/{@code tf.squeeze} producer registrations and the callee-return descent for
   * layer-call results add the degraded-rank members ({@code (D, D)}, {@code (D, D, D)}, {@code (8,
   * D, D)}): the einsum body's own reshapes now compute generator-side through the {@code
   * get_shape_list} walk, whose non-entry contexts resolve rank but not every dimension. The rank-4
   * {@code (8, 100, U, U)}/{@code (8, 10, U, U)} members are the {@code DenseLayer3dProj} contexts'
   * inputs (the attention's return value): the worklist engine converges the loop-carried union
   * from its non-cyclic base and all four proj contexts carry them (wala/ML#365 Phase 3 resolved
   * the fourth, the wala/ML#718 residual under the retired round-based resolution). The formerly
   * shape-⊤ members carry equation-proven ranks since the einsum-operand refinement (wala/ML#704):
   * {@code DenseLayer3d.call}'s {@code use_einsum} arm makes its input an operand of the rank-3
   * {@code "BFH"} term and {@code DenseLayer3dProj.call}'s of the rank-4 {@code "BFND"} term, and
   * the refined parameter states transport through the call boundary into this helper. Every proven
   * axis stays {@link UnresolvedDim} in vivo, since {@code w}'s extents are config-derived; the
   * union is dtype-homogeneous {@code float32} since the dtype feed (wala/ML#736) replaced the
   * attention path's pure-⊤ seeds. The {@code w} parameter keeps rank 3 and {@code float32} (its
   * chain is layer-local) but no numeric dimensions, since the {@code build}-computed head sizes
   * also derive from the config.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullEinsumViaMatmul()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "nlpgnn/layers/dense.py",
        "einsum_via_matmul",
        "nlpgnn_full_proj",
        2,
        14,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)),
                new TensorType(FLOAT_32, asList(new NumericDim(8), UnresolvedDim.INSTANCE)),
                new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(8), UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(8), new NumericDim(10), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(8), new NumericDim(100), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(8),
                        new NumericDim(10),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(8),
                        new NumericDim(100),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE))),
            3,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new SymbolicDim("?"), new SymbolicDim("?"), new SymbolicDim("?"))))));
  }

  /**
   * In-vivo anchor for wala/ML#736 at its measurement point: the vendored NLPGNN {@code
   * DenseLayer3d.call}'s {@code input_tensor}, formerly carrying an {@code unknown}-dtype member
   * alongside its {@code float32} ones (the heterogeneous union the issue reported). The dtype feed
   * replaces pure-⊤ generator seeds with synthetic edges from their dtype-source operands, so the
   * attention path's element-wise and pass-through results take the converged {@code float32}
   * instead of seeding {@code unknown}, and the union is dtype-homogeneous in every context. The
   * {@code (8, 100)}/{@code (8, 10)} leading pairs are the entry contracts (wala/ML#717); the ranks
   * are the einsum operand refinement's (wala/ML#704).
   *
   * <p>TODO: Drop the rank-4 {@code (8, 100, ?, ?)}/{@code (8, 10, ?, ?)} members once <a
   * href="https://github.com/wala/ML/issues/746">wala/ML#746</a>'s arm filtering reaches the
   * argument reads' points-to stage; the members ride the sibling {@code DenseLayer3dProj}'s
   * einsum-refined operands through the shared helper's conditional-reshape φ, whose infeasible arm
   * the points-to union cannot distinguish (the cross-context half was separated by the caller-node
   * context propagation).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullDense3dInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "nlpgnn/layers/dense.py",
        "DenseLayer3d.call",
        "nlpgnn_full_proj",
        1,
        9,
        Map.of(
            3,
            Set.of(
                new TensorType(FLOAT_32, asList(new NumericDim(8), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(8), new NumericDim(100), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(8), new NumericDim(10), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
                new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(8), UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(8),
                        new NumericDim(100),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(8),
                        new NumericDim(10),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)))));
  }

  /**
   * In-vivo anchor for the wala/ML#704 einsum-operand refinement at its reopen measurement point:
   * the vendored NLPGNN {@code DenseLayer3dProj.call}'s {@code input_tensor}, formerly {@code ? of
   * float32} (shape ⊤) in every context. The {@code use_einsum} arm's {@code
   * einsum("BFND,NDH->BFH", input_tensor, w)} proves the input rank 4; the shared {@code N}/{@code
   * D} extents stay {@link UnresolvedDim} in vivo because {@code w}'s dimensions are config-derived
   * (contrast {@link #testDense3dProj()}, where the weight's static extents transfer). The {@code
   * (8, 100, U, U)}/{@code (8, 10, U, U)} members arrive already rank-4 from the entry contracts
   * (wala/ML#717) and pass through the refinement with their concrete leading dims intact.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullDenseProjInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "nlpgnn/layers/dense.py",
        "DenseLayer3dProj.call",
        "nlpgnn_full_proj",
        1,
        8,
        Map.of(
            3,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(8),
                        new NumericDim(100),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(8),
                        new NumericDim(10),
                        UnresolvedDim.INSTANCE,
                        UnresolvedDim.INSTANCE)))));
  }

  /**
   * In-vivo anchor for the wala/ML#716 exactness mode and the wala/ML#717 contract seed: the
   * vendored NLPGNN {@code WDEmbedding.call}'s {@code input_ids} parameter resolves to each entry
   * script's declared {@code model.build(input_shape=(3, batch_size, maxlen))} contract, delivered
   * through the {@code tf.split}/{@code tf.squeeze}/{@code tf.cast} chain of the wrapper's {@code
   * call}. The expected set is the union across the vendored entry scripts' contracts (the helper
   * unions per value number across calling contexts), each contract's leading stack dimension
   * divided out by the split and squeezed away, leaving the per-entry {@code (batch_size, maxlen)}
   * pairs; all are {@code int32} after the cast.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullEmbeddingInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "nlpgnn/layers/embedding.py",
        "WDEmbedding.call",
        "nlpgnn_full_proj",
        1,
        12,
        Map.of(
            3,
            Set.of(
                TensorType.of(INT_32, 8, 100),
                TensorType.of(INT_32, 16, 100),
                TensorType.of(INT_32, 8, 10),
                TensorType.of(INT_32, 1, 512),
                TensorType.of(INT_32, 6, 128),
                TensorType.of(INT_32, 2, 4),
                TensorType.of(INT_32, 2, 10))));
  }

  /**
   * The first definition of the redefined top-level function of <a
   * href="https://github.com/wala/ML/issues/719">wala/ML#719</a>: the script defines {@code
   * compute} twice and calls each definition. Definition-site-distinct synthetic classes keep both
   * bodies, the module binding rebinds at the second {@code def}, and straight-line SSA binds each
   * call to the definition in scope, so this definition's parameter carries exactly its own call's
   * {@code (2, 2)} — where the pre-fix collapse lost this body entirely and bound its call to the
   * second definition.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRedefinedFunction()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_redefined_function.py",
        "compute",
        1,
        2,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2))));
  }

  /**
   * The second definition of {@link #testRedefinedFunction()}'s redefined function (wala/ML#719):
   * the definition at line 11 composes the position-disambiguated synthetic class {@code
   * compute$11}, and its parameter carries exactly the second call's {@code (3, 3)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRedefinedFunction2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_redefined_function.py",
        "compute$11",
        1,
        2,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 3))));
  }

  /**
   * Sibling half of {@link #testNlpgnnFullGeneration()} (wala/ML#690) — the {@code interactive}
   * entry script, the one whose {@code predict}/{@code call} nodes vanished in the consumer's
   * whole-project run. Its {@code predict} parameter typing must be symmetric with the generation
   * sibling's.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullInteractive()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "tests/TG/EN/interactive.py",
        "GenGPT2.predict",
        "nlpgnn_full_proj",
        1,
        1,
        Map.of(
            3, Set.of(new TensorType(UNKNOWN, asList(UnresolvedDim.INSTANCE, new NumericDim(1))))));
  }

  /**
   * Pins the <a href="https://github.com/wala/ML/issues/676">wala/ML#676</a> subject: {@code
   * DynamicPositionEmbedding.call}'s {@code inputs} parameter on the vendored {@code
   * jason9693/MusicTransformer-tensorflow2.0} {@code custom/layers.py}. With {@code
   * tf.keras.layers.Embedding} modeled and constructor keyword arguments forwarded (wala/ML#664),
   * the chain composes concretely: tokens {@code (2, 50)} int32 through the embedding to {@code (2,
   * 50, 64)} float32 into the position encoding. On 0.52.13 the parameter was tensor- classified
   * but shapeless ({@code ? of unknown}, an integer-arithmetic seed leaking through the pointer
   * analysis); wala/ML#664 removed the leak and this modeling supplies the honest chain.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   *     <p>The local count excludes the two all-constant slice-constructor objects under the
   *     multi-dim subscript, pinned non-tensor since wala/ML#732; the subscript result stays typed.
   */
  @Test
  public void testMusicTransformerPositionEmbedding()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "musictx_proj/custom/__init__.py",
          "musictx_proj/custom/layers.py",
          "musictx_proj/tf2_test_musictx_encoder.py"
        },
        "custom/layers.py",
        "DynamicPositionEmbedding.call",
        "musictx_proj",
        1,
        4,
        Map.of(3, Set.of(TensorType.of(FLOAT_32, 2, 50, 64))));
  }

  /**
   * {@code replica_fn(input)} body: {@code return input * 2.0}. Both {@code input} (parameter) and
   * the binop result are tensors, so the expected count is 2 (1 param + 1 binop-result SSA value).
   * Prior to wala/ML#395's scalar-literal-broadcast fix, the binop result was under-classified
   * (null shape) and didn't register, producing a count of 1. The updated count reflects the
   * corrected identification.
   */
  @Test
  public void testCallbacks()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_callbacks.py", "replica_fn", 1, 2, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** See {@link #testCallbacks()} for the count rationale. */
  @Test
  public void testCallbacks2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_callbacks2.py", "replica_fn", 1, 2, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Positive regression guard for chained-binop shape inference. The Python expression {@code (x +
   * y) * z} produces two nested {@code SSABinaryOpInstruction}s; shape inference for the outer
   * binop's inner-binop operand works via {@code ElementWiseOperation}'s recursive nested dispatch
   * (see {@code getOperandShapes} and wala/ML#395). If that recursive dispatch ever regresses, the
   * outer binop's operand shape lookup will fall to ⊤ and this test will fail.
   *
   * <p>Unrelated to wala/ML#398, which concerns PA-level allocation tracking for binop results (a
   * different failure mode that only manifests when a binop result flows through a PA-mediated
   * mechanism such as a field store).
   */
  @Test
  public void testChainedBinop()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_chained_binop.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Positive regression guard for binop-to-tuple-literal-to-subscript shape propagation. A Python
   * binop result {@code c = a + b} is stored into a tuple literal {@code (c,)} and read back via
   * subscript {@code t[0]}. The element type flows through SSA tracing (tuple-subscript dispatch
   * inspects the tuple's construction expression to resolve the element VN, then queries that VN's
   * tensor type via the standard generator path).
   *
   * <p>Notably, this works despite the binop producing no PA allocation — the tuple's field-0 PTS
   * is empty today, but the SSA-level path bypasses the PA field lookup. This is a different
   * recovery mechanism from {@code testChainedBinop}'s (recursive binop dispatch inside {@code
   * ElementWiseOperation}); both are worth guarding independently.
   *
   * <p>Contrasts with wala/ML#398: framework consumers like {@code
   * DatasetFromTensorSlicesGenerator} commit to PA field reads rather than SSA tracing, which is
   * where the binop-no-allocation gap actually bites. A test isolating that gap needs to route
   * through the dataset machinery.
   */
  @Test
  public void testBinopTupleStore()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_binop_tuple_store.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-only parameters are modeled as formal parameters (<a
   * href="https://github.com/wala/ML/issues/596">wala/ML#596</a>). {@code f(x, *, y)} is called
   * {@code f(tf.constant(1), y=tf.ones([2, 3]))}; the keyword-only {@code y} must be a formal so
   * the call-site keyword argument binds to it. {@code consume(y)} pins {@code y}'s type, which is
   * therefore {@code (2, 3) float32}. Before the fix, {@code y} had no value number and {@code
   * consume} saw no tensor parameter.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testKwonlyParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_kwonly_param.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Isolated repro for wala/ML#398 (binop drops PA allocation, bites through dataset). Python
   * {@code c = a + b; from_tensor_slices((c, y)); for x, _ in ds: consume(x)} — the binop result
   * {@code c} has no PA allocation and the tuple's field-0 PTS is empty. Passes without allocation
   * synthesis because {@link DatasetFromTensorSlicesGenerator#getShapesForIndex} and its dtype
   * counterpart now fall back to the SSA-chain helper on {@link TensorGenerator}, which walks the
   * DU from the tuple putfield's stored vn back to the concrete creator. See wala/WALA#1889 for the
   * upstream root-cause fix that would materialise PTS at synthetic-method return keys and make
   * this fallback unnecessary.
   */
  @Test
  public void testBinopThroughDataset()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_iso_binop_ds.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Regression guard for {@code tf.distribute.MirroredStrategy.distribute_datasets_from_function}'s
   * callback registration (wala/ML#113). The fixture registers {@code dataset_fn} only via {@code
   * strategy.distribute_datasets_from_function(dataset_fn)} &mdash; no other call site. The {@code
   * test(...)} helper's "function must exist in call graph" assertion fails pre-fix because {@code
   * dataset_fn} never gets traced. After this PR's `tensorflow.xml` fix (synthetic-method body on
   * {@code tensorflow/distribute/run/distribute_datasets_from_function} that allocates a stub
   * {@code InputContext} and invokes {@code dataset_fn(ctx)}), the callback enters the call graph
   * and the helper finds it. {@code input_context} is non-tensor, hence 0 tensor parameters; the 6
   * tensor variables in the body come from the chained {@code tf.data.Dataset} calls.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCallbacks3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_callbacks3.py", "dataset_fn", 0, 6, Map.of());
  }

  /**
   * {@code train_step(images, generator, discriminator, ...)} receives {@code image_batch} from the
   * training loop inside {@code train}. {@code image_batch} comes from iterating a dataset built
   * from mnist data via {@code train_images[..., None].astype(np.float32)} and {@code
   * from_tensor_slices(...).shuffle(...).batch(256)}. At runtime {@code image_batch} has shape in
   * {@code {(256, 28, 28, 1), (96, 28, 28, 1)}} dtype {@code float32} (60000 training images / 256
   * = 234 full batches + 1 partial batch of 96; verified by Python assert statements in {@code
   * tensorflow_gan_tutorial.py}).
   *
   * <p>Expected tensor variable count: 7. After wala/ML#430's {@code Gradient} generator allocates
   * a fresh tensor per {@code tape.gradient(...)} call instead of aliasing {@code sources}, each of
   * the two gradient calls in {@code train_step} (one for the generator, one for the discriminator)
   * registers an additional local tensor variable, lifting the count from the prior master baseline
   * of 5 to 7. Value 2 ({@code images}) is inferred concretely as {@code {(256, 28, 28, 1), (96,
   * 28, 28, 1)} float32}: the mnist pipeline resolves end to end. {@code mnist.load_data()}
   * supplies {@code (60000, 28, 28)} (<a
   * href="https://github.com/wala/ML/issues/361">wala/ML#361</a>), {@code [..., None]} and the
   * {@code (x - 127.5) / 127.5} binop chain carry it to {@code (60000, 28, 28, 1)}, {@code
   * from_tensor_slices} takes the element shape {@code (28, 28, 1)}, and {@code .batch(256)}
   * produces the two batch shapes (<a
   * href="https://github.com/wala/ML/issues/356">wala/ML#356</a>).
   */
  @Test
  public void testGanTutorial()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_gan_tutorial.py",
        "train_step",
        1,
        7,
        Map.of(2, Set.of(TENSOR_256_28_28_1_FLOAT32, TENSOR_96_28_28_1_FLOAT32)));
  }

  /**
   * Same structure as {@link #testGanTutorial()} but with {@code @tf.function} applied to {@code
   * train_step}. Runtime types for {@code image_batch} are identical and are verified via the
   * Python assert statements in {@code tensorflow_gan_tutorial.py} (not duplicated in {@code
   * tensorflow_gan_tutorial2.py} since the two files are structurally identical apart from the
   * decorator): shape in {@code {(256, 28, 28, 1), (96, 28, 28, 1)}}, dtype {@code float32}.
   * Expected count 7, same accounting as {@link #testGanTutorial()}: the two `tape.gradient(...)`
   * calls each contribute one fresh tensor variable post-wala/ML#430 (5 to 7). Value 2 is inferred
   * concretely as in {@link #testGanTutorial()}; the mnist binop-chain pipeline resolves (<a
   * href="https://github.com/wala/ML/issues/356">wala/ML#356</a>, <a
   * href="https://github.com/wala/ML/issues/361">wala/ML#361</a>).
   */
  @Test
  public void testGanTutorial2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_gan_tutorial2.py",
        "train_step",
        1,
        7,
        Map.of(2, Set.of(TENSOR_256_28_28_1_FLOAT32, TENSOR_96_28_28_1_FLOAT32)));
  }

  /**
   * Pins {@code generator_loss(fake_output)}'s parameter type. {@code fake_output} flows from
   * {@code discriminator(generated_images, ...)}; runtime shape is {@code (batch, 1) float32} since
   * the discriminator ends with a {@code Dense(1)} layer.
   *
   * <p>Inferred as {@code float32} with shape {@code ⊤} (unknown). The dtype is concrete and sound.
   * The shape is unknown rather than wrong: previously {@code ModelCall.getDefaultShapes} fell back
   * to the call's <em>input</em> shape when no output generator resolved, emitting the unsound
   * {@code (None, 100)} ({@code 100} is the generator's noise input dim from {@code
   * tf.keras.Input((100,))}, not the discriminator's {@code Dense(1)} output dim). A {@code
   * tf.keras.Model} generally transforms its input shape, so that fallback is removed
   * (wala/ML#537): the input shape now only refines a recovered output shape's batch dim, never
   * substitutes for it.
   *
   * <p>The runtime shape is {@code (256, 1)}/{@code (96, 1)} (verified by running the fixture:
   * {@code noise (256, 100)} -&gt; generator {@code (256, 28, 28, 1)} -&gt; discriminator {@code
   * (256, 1)}), confirming the old {@code (None, 100)} was the noise shape leaking through, not the
   * discriminator output.
   *
   * <p>TODO: tighten to {@code {(256, 1) float32, (96, 1) float32}}. The output node <em>is</em>
   * reachable now (the {@code outputs} construction argument points at the {@code Dense(1)} call),
   * but that {@code DenseCall} returns {@code null} shapes here because its input — the {@code
   * Flatten} of a {@code Conv2D} chain — isn't shape-tracked; recovering {@code (batch, 1)} needs
   * {@code DenseCall} output-shape inference through chained layer calls (tracked by <a
   * href="https://github.com/wala/ML/issues/358">wala/ML#358</a>), which in turn needs {@code
   * Conv2D}/{@code Flatten} output-shape modeling. Concrete batch dims 256/96 come from {@code
   * train_step}'s {@code images} parameter, per {@link #testGanTutorial}. (wala/ML#537 fixed the
   * unsound mis-propagation; this residual precision is wala/ML#358's domain.)
   */
  @Test
  public void testGanTutorialGeneratorLoss()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_gan_tutorial.py",
        "generator_loss",
        1,
        2,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Pins {@code top_p_logits(logits, p)}'s parameter type. Function body mirrors {@code
   * akanyaani/gpt-2-tensorflow2.0/sample.py}'s {@code top_p_logits}. Exercises several ops
   * currently routed through {@code ReadDataFallback} per wala/ML#449 ({@code tf.sort}, {@code
   * tf.cumsum}, {@code tf.stack}, {@code tf.range}, {@code tf.gather_nd}, {@code tf.where}), but
   * the parameter type of {@code logits} comes from its caller (a {@code tf.constant} with shape
   * {@code (1, 5)} dtype {@code float32}), so this test isolates caller-side propagation rather
   * than the body's op precision.
   *
   * <p>Empirically, {@code logits} is inferred as {@code (1, 5) float32} — concrete on both axes.
   * The caller's {@code tf.constant([[1.0, ..., 5.0]], dtype=tf.float32)} flows in cleanly, showing
   * that none of the body's {@code ReadDataFallback}-routed ops block caller-side propagation for
   * this function; the parameter type is fully resolved at the call site.
   */
  @Test
  public void testTopPLogits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_top_p_logits.py", "top_p_logits", 1, 12, Map.of(2, Set.of(TENSOR_1_5_FLOAT32)));
  }

  /**
   * Pins {@code _take_long_axis(arr, indices)}'s parameter types. Function body mirrors {@code
   * _take_long_axis} from {@code
   * LongmaoTeamTf/deep_recommenders/keras/models/retrieval/factorized_top_k.py}.
   *
   * <p>This fixture surfaced a real Ariadne bug: {@code tf.reshape(arr, tf.shape(other))} crashes
   * the analysis with {@code IllegalStateException} at {@code
   * TensorGenerator.getShapesFromShapeArgument} because the shape argument is a Tensor (the result
   * of {@code tf.shape(...)}) rather than a list/tuple literal. The {@code Reshape} generator
   * should gracefully degrade to ⊤ shape per the lattice conventions instead of throwing. Resolved
   * by <a href="https://github.com/wala/ML/issues/538">wala/ML#538</a>; parameter types pin
   * precisely.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/543">wala/ML#543</a>): the post-fix
   * local-tensor count of 9 captures every intermediate runtime-shape tensor flowing through the
   * body. Tighten once the body-level imprecision is addressed.
   */
  @Test
  public void testTakeAlongAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_take_along_axis.py",
        "_take_long_axis",
        2,
        10,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32), 3, Set.of(TENSOR_2_2_INT32)));
  }

  /**
   * Pins {@code multilayer_perceptron(x)}'s parameter type. Function body mirrors {@code
   * multilayer_perceptron} from {@code
   * YunYang1994/TensorFlow2.0-Examples/2-Basical_Models/Multilayer_Perceptron.py}. Uses raw {@code
   * tf.matmul} / {@code tf.add} / {@code tf.nn.sigmoid} / {@code tf.nn.softmax} against global
   * {@code tf.Variable} weights and biases — a different pattern from the {@code Dense}-layer
   * subclass-Model approach already covered by {@code testNeuralNetwork*}.
   *
   * <p>{@code x} is inferred as {@code (100, 784) float32}, flowing from the caller's {@code
   * batch_x = tf.constant(np.ones((100, 784), dtype=np.float32))}. This relies on the {@code numpy
   * → tf.constant} dtype/shape bridge fixed in <a
   * href="https://github.com/wala/ML/issues/539">wala/ML#539</a> (see {@link
   * #testConstantFromNumpy} for the isolated guard).
   *
   * <p>The local-tensor count is 16 (up from 14 before wala/ML#539): with {@code x} now typed, two
   * further {@code tf.matmul}/{@code tf.add} intermediates that consume it are recognized as
   * tensors.
   */
  @Test
  public void testMultilayerPerceptron()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_multilayer_perceptron.py",
        "multilayer_perceptron",
        1,
        16,
        Map.of(2, Set.of(TENSOR_100_784_FLOAT32)));
  }

  /**
   * Pins {@code logistic_regression(x)}'s parameter type. Function body mirrors {@code
   * logistic_regression} from {@code
   * aymericdamien/TensorFlow-Examples/.../2_BasicModels/logistic_regression.py}, a real-world
   * image-classification utility (logistic regression: {@code softmax(W x + b)} over global {@code
   * tf.Variable} weights and biases), for tensor-type inference coverage. Like {@link
   * #testMultilayerPerceptron}, it uses raw {@code tf.matmul} / {@code tf.nn.softmax} rather than
   * the {@code Dense}-layer subclass-{@code Model} approach of {@code testNeuralNetwork*}.
   *
   * <p>{@code x} is inferred as {@code (100, 784) float32}—both shape and dtype concrete—flowing
   * from the caller's {@code tf.constant(np.ones((100, 784), dtype=np.float32))} via the {@code
   * numpy→tf.constant} bridge (<a href="https://github.com/wala/ML/issues/539">wala/ML#539</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLogisticRegression()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_logistic_regression.py",
        "logistic_regression",
        1,
        6,
        Map.of(2, Set.of(TENSOR_100_784_FLOAT32)));
  }

  /**
   * Pins {@code nce_loss(x_embed, y)}'s parameter types. Function body mirrors {@code nce_loss}
   * from {@code aymericdamien/TensorFlow-Examples/.../2_BasicModels/word2vec.py}, a real-world
   * word-embedding utility (the averaged noise-contrastive-estimation loss over global {@code
   * tf.Variable} embedding/weight/bias matrices), for tensor-type inference coverage.
   *
   * <p>Both parameters are inferred concretely—shape and dtype: {@code x_embed} as {@code (4, 10)
   * float32} and {@code y} as {@code (4, 1) int32}, flowing from the {@code
   * tf.constant(np.ones(...))} call site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNceLoss()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nce_loss.py",
        "nce_loss",
        2,
        5,
        Map.of(2, Set.of(TENSOR_4_10_FLOAT32), 3, Set.of(TENSOR_4_1_INT32)));
  }

  /**
   * Pins {@code evaluate(x_embed)}'s parameter type. Function body mirrors {@code evaluate} from
   * {@code aymericdamien/TensorFlow-Examples/.../2_BasicModels/word2vec.py}, a real-world
   * word-embedding utility (the cosine similarity between an input embedding and every row of the
   * global embedding matrix), for tensor-type inference coverage.
   *
   * <p>{@code x_embed} is inferred as {@code (4, 10) float32}—both shape and dtype concrete—flowing
   * from the {@code tf.constant(np.ones(...))} call site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEvaluate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_evaluate.py", "evaluate", 1, 13, Map.of(2, Set.of(TENSOR_4_10_FLOAT32)));
  }

  /**
   * Pins {@code random_jitter(input_image, real_image)}'s parameter types. Function body (and the
   * {@code resize}/{@code random_crop} helpers it calls) mirrors {@code random_jitter} from {@code
   * YunYang1994/TensorFlow2.0-Examples/.../Pix2Pix.py}, a real-world image-to-image translation
   * utility (random resize/crop/mirror data augmentation), for tensor-type inference coverage.
   *
   * <p>Both image parameters are inferred concretely—shape and dtype—as {@code (256, 256, 3)
   * float32}, flowing from the {@code tf.constant(np.ones(...))} call site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRandomJitter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_random_jitter.py",
        "random_jitter",
        2,
        5,
        Map.of(2, Set.of(TENSOR_256_256_3_FLOAT32), 3, Set.of(TENSOR_256_256_3_FLOAT32)));
  }

  /**
   * Pins {@code MaskSparseCategoricalCrossentropy.__call__(y_true, y_predict, input_mask)}'s
   * parameter types. Class and method body mirror {@code MaskSparseCategoricalCrossentropy} from
   * {@code kyzhouhzau/NLPGNN/nlpgnn/metrics/Losess.py}, a real-world NLP utility (a mask-weighted
   * sparse-categorical-crossentropy loss), for tensor-type inference coverage. Unlike the {@code
   * tf.keras.Model.call} layer-chain methods ({@code testNeuralNetwork*}), this is a loss {@code
   * __call__} on a plain class that reduces its inputs to a scalar.
   *
   * <p>All three parameters are inferred concretely—shape and dtype: {@code y_true} as {@code (4,)
   * int32}, {@code y_predict} as {@code (4, 10) float32}, and {@code input_mask} as {@code (4,)
   * float32}, flowing from the {@code tf.constant(np.ones(...))} call site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testMaskedSparseCrossentropy()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_masked_sparse_ce.py",
        "MaskSparseCategoricalCrossentropy.__call__",
        3,
        7,
        Map.of(
            3, Set.of(TENSOR_4_INT32),
            4, Set.of(TENSOR_4_10_FLOAT32),
            5, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * Pins {@code LSTM.call(x)}'s parameter type. Class and method body mirror the {@code LSTM}
   * recurrent model from {@code
   * aymericdamien/TensorFlow-Examples/.../3_NeuralNetworks/recurrent_network.py}, a real-world
   * sequence-classification utility (a built-in {@code tf.keras.layers.LSTM} followed by a {@code
   * Dense} read-out), for tensor-type inference coverage.
   *
   * <p>The input parameter {@code x} is recovered concretely on both axes—{@code (256, 28, 28)
   * float32}—flowing from the {@code lstm_net(x, is_training=True)} call site through {@code
   * tf.keras.Model.__call__} dispatch.
   *
   * <p>The forward-pass locals ({@code lstm_layer} output, {@code out} output, {@code softmax}) are
   * inferred as {@code float32} but with <em>unknown shape</em>: the built-in {@code LSTM}/{@code
   * Dense} output shapes are not narrowed (the layer-chain shape gap tracked by <a
   * href="https://github.com/wala/ML/issues/530">wala/ML#530</a>). The dtype axis—the load-bearing
   * one—is exact; only shape is ⊤.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLstmCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_lstm_call.py", "LSTM.call", 1, 4, Map.of(3, Set.of(TENSOR_256_28_28_FLOAT32)));
  }

  /**
   * Pins {@code GCNLayer.call(node_embeddings, adjacency_lists)}'s tensor-parameter type. The
   * {@code GCNLayer} model and the {@code GraphConvolution}/{@code MessagePassing} layers it builds
   * on are vendored verbatim from {@code kyzhouhzau/NLPGNN} ({@code nlpgnn/models/GCN.py}, {@code
   * nlpgnn/gnn/GCNConv.py}, {@code nlpgnn/gnn/messagepassing.py}); only the driver and a
   * reachable-slice {@code nlpgnn/gnn/utils.py} (just the {@code GNNInput} named tuple) are
   * bespoke. This is a real-world graph-neural-network utility (a two-layer graph-convolution
   * message-passing model), exercised for tensor-type inference coverage across a multi-module
   * import chain.
   *
   * <p>The tensor parameter {@code node_embeddings} is recovered concretely on both axes—{@code (4,
   * 8) float32}—flowing from the driver's {@code model(node_embeddings, adjacency_lists,
   * training=False)} call site through {@code tf.keras.Model.__call__} dispatch, across the {@code
   * driver→GCN→GCNConv→MessagePassing} module boundaries. ({@code adjacency_lists} is a Python list
   * of edge tensors, not a tensor itself, so it is not a tensor parameter.)
   *
   * <p>The message-passing <em>output</em> locals (the {@code gc1}/{@code gc2} results) are still
   * inferred as ⊤, but the cause is now downstream of the matmul rather than the {@code NamedTuple}
   * field read. The aggregation ops are modeled, and the {@code GNNInput} {@code NamedTuple} field
   * read {@code node_embeddings = inputs.node_embeddings} recovers {@code (4, 8) float32} (read off
   * the instance field in the heap; <a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a>), so {@code tf.linalg.matmul} and
   * {@code propagate} inside {@code GraphConvolution.call} now type to {@code float32}. What
   * remains is the forward-result hop: {@code GraphConvolution.call} returns a typed tensor, but
   * that result does not propagate to the caller's {@code self.gc1(...)} local &mdash; the
   * user-subclass forward-result-typing gap (wala/ML#570, akin to <a
   * href="https://github.com/wala/ML/issues/595">wala/ML#595</a>). The decorated function's input
   * signature, the analysis goal, is nonetheless exact. The returned {@code tf.math.softmax} result
   * counts as one more tracked variable since the {@code math.softmax} alias was wired
   * (wala/ML#711).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGcnCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gcn_proj/nlpgnn/__init__.py",
          "gcn_proj/nlpgnn/gnn/__init__.py",
          "gcn_proj/nlpgnn/gnn/utils.py",
          "gcn_proj/nlpgnn/gnn/messagepassing.py",
          "gcn_proj/nlpgnn/gnn/GCNConv.py",
          "gcn_proj/nlpgnn/models/__init__.py",
          "gcn_proj/nlpgnn/models/GCN.py",
          "gcn_proj/tf2_test_gcn_call.py"
        },
        "nlpgnn/models/GCN.py",
        "GCNLayer.call",
        "gcn_proj",
        1,
        7,
        Map.of(3, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins {@code GATLayer.call(node_embeddings, adjacency_lists)}'s tensor-parameter type. The
   * attention counterpart of {@link #testGcnCall()}: the {@code GATLayer} model and the {@code
   * GraphAttentionConvolution}/{@code MessagePassing} layers it builds on are vendored verbatim
   * from {@code kyzhouhzau/NLPGNN} ({@code nlpgnn/models/GAT.py}, {@code nlpgnn/gnn/GATConv.py},
   * {@code nlpgnn/gnn/messagepassing.py}); only the driver and a reachable-slice {@code
   * nlpgnn/gnn/utils.py} (the {@code GNNInput} named tuple plus {@code masksoftmax}/{@code
   * maybe_num_nodes}) are bespoke.
   *
   * <p>The tensor parameter {@code node_embeddings} is recovered concretely on both axes &mdash;
   * {@code (4, 8) float32} &mdash; flowing from the driver's {@code model(node_embeddings,
   * adjacency_lists, training=False)} call site through {@code tf.keras.Model.__call__} dispatch,
   * across the {@code driver→GAT→GATConv→MessagePassing} module boundaries. ({@code
   * adjacency_lists} is a Python list of edge tensors, not a tensor itself, so it is not a tensor
   * parameter.) As with {@link #testGcnCall()}, the decorated function's input signature &mdash;
   * the analysis goal &mdash; is exact, while the internal layer-output locals stay ⊤. The five
   * tracked tensor variables are {@code node_embeddings} (vn=3) and the {@code dropout1} output
   * (vn=11), both concrete {@code (4, 8) float32} (dropout preserves shape and dtype), the {@code
   * gc1}/{@code gc2} attention-convolution outputs (vn=18, vn=38), both ⊤ on each axis, plus the
   * returned {@code tf.math.softmax} result, typed since the {@code math.softmax} alias was wired
   * (wala/ML#711). The layer outputs are ⊤ for the same reason as GCN: {@code
   * GraphAttentionConvolution.call} unwraps its input from a {@code GNNInput} {@code NamedTuple}
   * field, which is not tracked as a tensor (wala/ML#579), so the input arrives ⊤ and the attention
   * aggregation through {@code tf.math.unsorted_segment_*} (wala/ML#570, wala/ML#582) inherits it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGatCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gat_proj/nlpgnn/__init__.py",
          "gat_proj/nlpgnn/gnn/__init__.py",
          "gat_proj/nlpgnn/gnn/utils.py",
          "gat_proj/nlpgnn/gnn/messagepassing.py",
          "gat_proj/nlpgnn/gnn/GATConv.py",
          "gat_proj/nlpgnn/models/__init__.py",
          "gat_proj/nlpgnn/models/GAT.py",
          "gat_proj/tf2_test_gat_call.py"
        },
        "nlpgnn/models/GAT.py",
        "GATLayer.call",
        "gat_proj",
        1,
        5,
        Map.of(3, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Captured-gap reproduction for <a href="https://github.com/wala/ML/issues/659">wala/ML#659</a>:
   * in the vendored GAT subject, {@code maybe_num_nodes}'s {@code index} parameter (vn=2) is
   * currently <em>not</em> typed as a tensor, even though it receives a real tensor.
   *
   * <p>The chain: {@code MessagePassing._calculate_messages_all_type} computes {@code edge_targets
   * = adjanceny_list_edge_type[:, 1]} (a subscript-slice of an {@code enumerate(adjacency_lists)}
   * element), which flows through {@code GATConv.message_function}'s {@code edge_target} into
   * {@code masksoftmax(alpha, edge_target)} and then {@code maybe_num_nodes(index, ...)}. In {@code
   * propagate} the adjacency list types correctly to {@code (E, 2) int32}, but passed into {@code
   * _calculate_messages_all_type} the {@code adjacency_lists} parameter is context-collapsed on the
   * shared message-passing summary (merged with the {@code float32} {@code node_embeddings}),
   * losing its {@code int32}/rank-2 type. The corrupted ⊤ makes the {@code enumerate} element
   * empty-shaped, so the {@code [:, 1]} subscript returns ⊥ and {@code index} never types. This is
   * a regression from 0.52.9 (bisected to the {@code SliceBuiltinOperation} empty-shape change in
   * wala/ML#656, which unmasked the pre-existing collapse; ⊤ before, ⊥ after), and the root cause
   * is the same context-collapse class as <a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a>.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/659">wala/ML#659</a>): {@code index} should
   * type as a tensor once the shared-summary context collapse is resolved. Flip this to assert vn=2
   * <em>is</em> typed when it is fixed.
   */
  @Test
  public void testGatMaybeNumNodesIndexUntyped()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(
            getPathFiles("gat_proj"),
            new String[] {
              "gat_proj/nlpgnn/__init__.py",
              "gat_proj/nlpgnn/gnn/__init__.py",
              "gat_proj/nlpgnn/gnn/utils.py",
              "gat_proj/nlpgnn/gnn/messagepassing.py",
              "gat_proj/nlpgnn/gnn/GATConv.py",
              "gat_proj/nlpgnn/models/__init__.py",
              "gat_proj/nlpgnn/models/GAT.py",
              "gat_proj/tf2_test_gat_call.py"
            });
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    // Guard against a vacuous pass: the captured-gap assertion below only checks that vn=2 is
    // *absent* from the typed set, which is trivially satisfied if `maybe_num_nodes` is no longer
    // reached at all (e.g. the entrypoint or file list changes). Require a reachable node so the
    // test fails, rather than passing silently, when the reproduction stops exercising the target.
    assertTrue(
        "The reproduction must reach `maybe_num_nodes`; otherwise this captured-gap guard passes"
            + " vacuously (wala/ML#659).",
        CG.stream().anyMatch(n -> n.getMethod().getSignature().contains("maybe_num_nodes")));

    // Collect the parameter value numbers that `maybe_num_nodes` types as tensors. Its `index`
    // parameter is vn=2 (vn=1 is the function object).
    //
    // Coverage note: while the wala/ML#659 gap holds, no `maybe_num_nodes` parameter types, so the
    // `isParameter() && ...maybe_num_nodes` branch is never taken and `typedParamVns.add(...)` is
    // never reached (Codecov reports them as an uncovered line and partial branch). That is the
    // captured gap itself; both become covered when #659 is fixed and the assertion below is
    // flipped
    // to expect vn=2 typed.
    Set<Integer> typedParamVns = new HashSet<>();
    analysis.forEach(
        pt -> {
          if (pt.fst instanceof LocalPointerKey) {
            LocalPointerKey lpk = (LocalPointerKey) pt.fst;
            if (lpk.isParameter()
                && lpk.getNode().getMethod().getSignature().contains("maybe_num_nodes"))
              typedParamVns.add(lpk.getValueNumber());
          }
        });
    assertFalse(
        "Captured gap for wala/ML#659: `maybe_num_nodes`'s `index` (vn=2) is currently untyped"
            + " because `adjacency_lists` is context-collapsed on the shared message-passing"
            + " summary. Flip this assertion when the collapse is resolved.",
        typedParamVns.contains(2));
  }

  /**
   * Pins {@code GCN.call(self, features, adj)}'s tensor-parameter types. The {@code GCN} layer is
   * vendored verbatim from {@code LongmaoTeamTf/deep_recommenders} ({@code
   * deep_recommenders/keras/models/retrieval/gcn.py}); only the driver is bespoke. This is a
   * real-world graph-convolution layer, exercised for tensor-type inference coverage of a
   * <em>sparse tensor parameter</em>: unlike the other vendored layer methods, {@code adj} is a
   * {@code tf.SparseTensor}, on which the layer branches ({@code isinstance(adj, tf.SparseTensor)})
   * to use {@code tf.sparse.sparse_dense_matmul}. The driver feeds a sparse {@code adj}, mirroring
   * {@code train_gcn_on_cora_keras.py}'s {@code GCN(32)(feats, g)} call site where {@code g} is a
   * {@code scipy.sparse} adjacency.
   *
   * <p>Both parameters are recovered concretely: {@code features} (vn=3) as {@code (4, 8) float32}
   * and the sparse {@code adj} (vn=4) as {@code (4, 4) float32} with {@link
   * com.ibm.wala.cast.python.ml.types.TensorType.Layout#SPARSE} layout (a {@link
   * SparseTensorType}). So the sparse parameter is typed precisely, including its sparse layout,
   * rather than collapsing to dense or ⊤ &mdash; the sparse {@code TensorType} representation of <a
   * href="https://github.com/wala/ML/issues/588">wala/ML#588</a>. (The {@code **kwargs} parameter
   * is not a tensor and is not extracted.) Emitting {@code tf.SparseTensorSpec} for such a
   * parameter in an inferred signature is the downstream consumer's job; this confirms the typed
   * input it needs is available.
   *
   * <p>The local-tensor count rose from 4 to 7 when the Keras lazy-{@code build} protocol was
   * modeled (<a href="https://github.com/wala/ML/issues/595">wala/ML#595</a>): {@code self._kernel}
   * now dispatches, so {@code outputs} (the {@code Dense} result), the residual {@code outputs +=
   * features} value, and the return alias gained tensor types &mdash; a precision gain.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGcnSparseCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gcn_sparse_proj/deep_recommenders/__init__.py",
          "gcn_sparse_proj/deep_recommenders/keras/__init__.py",
          "gcn_sparse_proj/deep_recommenders/keras/models/__init__.py",
          "gcn_sparse_proj/deep_recommenders/keras/models/retrieval/__init__.py",
          "gcn_sparse_proj/deep_recommenders/keras/models/retrieval/gcn.py",
          "gcn_sparse_proj/tf2_test_gcn_sparse_call.py"
        },
        "deep_recommenders/keras/models/retrieval/gcn.py",
        "GCN.call",
        "gcn_sparse_proj",
        2,
        7,
        Map.of(3, Set.of(TENSOR_4_8_FLOAT32), 4, Set.of(SPARSE_TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the chained-layer forward result (<a href="https://github.com/wala/ML/issues/595">
   * wala/ML#595</a>): a value bound to a user-defined Keras {@code Layer}'s call is tensor-typed at
   * the call site, so it flows as a tensor into a downstream function. The fixture chains two
   * {@code GCN} layers (vendored from {@code deep_recommenders}), mirroring {@code
   * train_gcn_on_cora_keras.py}'s {@code x = GCN(32)(feats, g); GCN(num_classes)(x, g)}, and sinks
   * the first layer's output through {@code consume(hidden)}.
   *
   * <p>{@code hidden} (the {@code GCN.call} return, a {@code Dense} output) is tensor-classified
   * because the modeled Keras lazy-{@code build} protocol invokes {@code GCN.build}, giving the
   * {@code build()}-created {@code self._kernel} sublayer a points-to set. With constructor keyword
   * arguments forwarded to {@code __init__} (wala/ML#664) and the layer-method trampolines keyed on
   * the receiver instance (wala/ML#679), each instance's {@code build} constructs its own {@code
   * Dense}, so the runtime-true {@code (4, 16) float32} is inferred without the other instance's
   * spurious {@code (4, 8)}. The remaining {@code ? of float32} member comes from the statically
   * dead {@code outputs += features} residual branch ({@code self._residual} is constantly {@code
   * False}), which the path-insensitive analysis still evaluates.
   *
   * <p>TODO: Expect exactly {@code (4, 16) float32} once <a
   * href="https://github.com/wala/ML/issues/681">wala/ML#681</a> prunes branches guarded by
   * statically-constant instance fields.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGcnChainConsume()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gcn_chain_proj/deep_recommenders/__init__.py",
          "gcn_chain_proj/deep_recommenders/keras/__init__.py",
          "gcn_chain_proj/deep_recommenders/keras/models/__init__.py",
          "gcn_chain_proj/deep_recommenders/keras/models/retrieval/__init__.py",
          "gcn_chain_proj/deep_recommenders/keras/models/retrieval/gcn.py",
          "gcn_chain_proj/tf2_test_gcn_chain_call.py"
        },
        "tf2_test_gcn_chain_call.py",
        "consume",
        "gcn_chain_proj",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 16), TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Pins {@code TextCNN.call(inputs, training)}'s tensor-parameter type. The {@code TextCNN} model
   * is vendored verbatim from {@code kyzhouhzau/NLPGNN} ({@code nlpgnn/models/TextCNN.py}); only
   * the driver is bespoke. This is a real-world text-classification utility (a convolutional
   * sentence encoder: embedding lookup, parallel {@code Conv1D} kernels, global average pooling,
   * concatenation, batch normalization, and a softmax dense head), exercised for tensor-type
   * inference coverage. Unlike the GNN cohort ({@link #testGcnCall()}, {@link #testGatCall()}) and
   * the float-feature method archetypes, the decorated parameter {@code inputs} is an integer
   * token-ID tensor (an embedding-lookup index), so this measures int-dtype parameter recovery.
   *
   * <p>The tensor parameter {@code inputs} is recovered concretely on both axes &mdash; {@code (2,
   * 5) int32} &mdash; flowing from the driver's {@code model(inputs, training=False)} call site
   * through {@code tf.keras.Model.__call__} dispatch into {@code TextCNN.call}. As with the GNN
   * cohort, the decorated function's input signature &mdash; the analysis goal &mdash; is exact;
   * here it confirms that recovery holds for an integer-dtype parameter and a convolutional
   * (non-message-passing) body, not only the float-feature archetypes.
   *
   * <p>The tracked tensor variables are {@code inputs} (concrete {@code (2, 5) int32}), the
   * embedding output (concrete {@code (2, 5, 8)} float32 with {@code Embedding} modeled,
   * wala/ML#676), and the softmax {@code Dense} head's output ({@code float32} dtype but ⊤ shape
   * &mdash; {@code DenseCall} hard-codes {@code float32} but loses the shape through the
   * chained-layer body, <a href="https://github.com/wala/ML/issues/358">wala/ML#358</a>/<a
   * href="https://github.com/wala/ML/issues/530">wala/ML#530</a>). The convolution intermediate is
   * ⊥: {@code Conv1D}/{@code GlobalAvgPool1D} remain unmodeled, and its former both-axes-⊤ rode the
   * spurious unknown-tensor composition of the enumerate over the unresolved {@code kernel_sizes}
   * list, which wala/ML#730 removed. These residual body locals are pre-existing modeling gaps, not
   * new findings, and are downstream of the (exact) input signature.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTextcnnCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "textcnn_proj/nlpgnn/__init__.py",
          "textcnn_proj/nlpgnn/models/__init__.py",
          "textcnn_proj/nlpgnn/models/TextCNN.py",
          "textcnn_proj/tf2_test_textcnn_call.py"
        },
        "nlpgnn/models/TextCNN.py",
        "TextCNN.call",
        "textcnn_proj",
        1,
        3,
        Map.of(3, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Pins the output type of {@code tf.math.unsorted_segment_sum} (wala/ML#570). The output dtype
   * inherits from the {@code data} input ({@code float32}); the shape is the concrete {@code (2,
   * 3)} — {@code [num_segments] ++ data.shape[segment_ids.ndim:]} with the static {@code
   * num_segments = 2}, rank-1 {@code segment_ids}, and {@code (3, 3)} {@code data} (wala/ML#582).
   * Verified via a {@code consume_sum} sink on the aggregation result.
   */
  @Test
  public void testUnsortedSegmentSum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_unsorted_segment.py", "consume_sum", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins the output type of {@code tf.math.unsorted_segment_max} (wala/ML#570). Same dtype-from-
   * {@code data} and static-{@code num_segments} {@code (2, 3)} shape recovery as {@link
   * #testUnsortedSegmentSum} (wala/ML#582).
   */
  @Test
  public void testUnsortedSegmentMax()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_unsorted_segment.py", "consume_max", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins the output type of {@code tf.math.unsorted_segment_mean} (wala/ML#570). Same dtype-from-
   * {@code data} and static-{@code num_segments} {@code (2, 3)} shape recovery as {@link
   * #testUnsortedSegmentSum} (wala/ML#582).
   */
  @Test
  public void testUnsortedSegmentMean()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_unsorted_segment.py",
        "consume_mean",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins {@code Transformer.call(encoder_inputs, decoder_inputs)}'s parameter types. The {@code
   * Transformer} layer and the {@code MultiHeadAttention}/{@code ScaledDotProductAttention} layers
   * it builds on are vendored verbatim from {@code LongmaoTeamTf/deep_recommenders} ({@code
   * deep_recommenders/keras/models/nlp/transformer.py} and {@code .../multi_head_attention.py});
   * only the driver is bespoke. This is a real-world sequence-to-sequence utility (a
   * multi-head-attention encoder/decoder transformer), exercised for tensor-type inference coverage
   * across a multi-module import chain.
   *
   * <p>Both token-sequence parameters are recovered concretely on both axes—{@code (2, 5) int32}
   * each—flowing from the driver's {@code transformer(encoder_inputs, decoder_inputs)} call site
   * through {@code tf.keras.layers.Layer.__call__} dispatch, across the {@code
   * deep_recommenders→nlp→transformer} package boundaries.
   *
   * <p>With {@code tf.keras.backend} modeled (<a
   * href="https://github.com/wala/ML/issues/666">wala/ML#666</a>), the padding mask {@code masks =
   * K.equal(inputs, 0)} is a third function-local tensor, typed {@code (2, 5)} bool. With {@code
   * add_weight} consuming its arguments (wala/ML#667), {@code embeddings =
   * K.gather(self.embeddings, inputs)} is a fourth: the embedding table types {@code (?, 8)}
   * float32 and the {@code gather} pass-through carries it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTransformerCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "tr_proj/deep_recommenders/__init__.py",
          "tr_proj/deep_recommenders/keras/__init__.py",
          "tr_proj/deep_recommenders/keras/models/__init__.py",
          "tr_proj/deep_recommenders/keras/models/nlp/__init__.py",
          "tr_proj/deep_recommenders/keras/models/nlp/multi_head_attention.py",
          "tr_proj/deep_recommenders/keras/models/nlp/transformer.py",
          "tr_proj/tf2_test_transformer_call.py"
        },
        "deep_recommenders/keras/models/nlp/transformer.py",
        "Transformer.call",
        "tr_proj",
        2,
        4,
        Map.of(3, Set.of(TENSOR_2_5_INT32), 4, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Pins {@code crf_unary_score(tag_indices, sequence_lengths, inputs)}'s parameter types. Function
   * body mirrors {@code crf_unary_score} from {@code kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py}, a
   * real-world linear-chain CRF function exercised for tensor-type inference coverage. Its {@code
   * tf.reshape} with runtime-derived dimensions ({@code tf.shape(inputs)[0]}) previously crashed
   * the analysis (wala/ML#567). The local count includes the {@code tf.range(...) * shape-element}
   * offset chain, whose rank-0 co-operands preserve the range results' shapes through the
   * elementwise scalar rule (wala/ML#723).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfUnaryScore()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_unary_score",
        3,
        22,
        Map.of(
            2, Set.of(TENSOR_2_3_INT32),
            3, Set.of(TENSOR_2_INT32),
            4, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * Pins {@code crf_binary_score(tag_indices, sequence_lengths, transition_params)}'s parameter
   * types. Function body mirrors {@code crf_binary_score} from {@code
   * kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py} for tensor-type inference coverage.
   *
   * <p>The local count includes the flat-index arithmetic ({@code start_tag_indices * num_tags +
   * end_tag_indices}), typed exactly {@code (2, 2) int32}: a {@code tf.shape} element is a rank-0
   * tensor, so the elementwise scalar-co-operand rule preserves the tensor operand's shape through
   * the nested product (wala/ML#723).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfBinaryScore()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_binary_score",
        3,
        14,
        Map.of(
            2, Set.of(TENSOR_2_3_INT32),
            3, Set.of(TENSOR_2_INT32),
            4, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins {@code crf_sequence_score(inputs, tag_indices, sequence_lengths, transition_params)}'s
   * parameter types. Function body mirrors {@code crf_sequence_score} from {@code
   * kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py} for tensor-type inference coverage.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfSequenceScore()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_sequence_score",
        4,
        9,
        Map.of(
            2, Set.of(TENSOR_2_3_4_FLOAT32),
            3, Set.of(TENSOR_2_3_INT32),
            4, Set.of(TENSOR_2_INT32),
            5, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins {@code crf_log_norm(inputs, sequence_lengths, transition_params)}'s parameter types.
   * Function body mirrors {@code crf_log_norm} from {@code kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py}
   * for tensor-type inference coverage.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfLogNorm()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_log_norm",
        3,
        9,
        Map.of(
            2, Set.of(TENSOR_2_3_4_FLOAT32),
            3, Set.of(TENSOR_2_INT32),
            4, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins {@code crf_forward(inputs, state, transition_params, sequence_lengths)}'s parameter types.
   * Function body mirrors {@code crf_forward} from {@code kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py}
   * for tensor-type inference coverage. {@code crf_forward} is reached only through {@code
   * crf_log_norm} (its sole NLPGNN caller), which passes {@code inputs} and {@code state} from
   * {@code tf.slice}/{@code tf.squeeze} results over a {@code (2, 3, 4)} constant. {@code inputs}
   * (from {@code tf.slice(inputs, [0, 1, 0], [-1, -1, -1])}) infers as {@code (2, 2, 4)} via the
   * {@code begin}/{@code size} shape derivation (wala/ML#569). {@code state} (from {@code
   * tf.squeeze(tf.slice(inputs, [0, 0, 0], [-1, 1, -1]), [1])}) infers as {@code (2, 4)}: the
   * {@code Slice} shape gives {@code (2, 1, 4)} and {@code tf.squeeze}'s axis-1 removal
   * (wala/ML#513) drops the singleton. Both were previously {@code ⊤}-shaped (dtype only,
   * wala/ML#568).
   *
   * <p>The local tensor-variable count is {@code 15}: the two {@code tf.transpose} calls now
   * allocate distinct tensors rather than aliasing their inputs as first-argument {@code
   * pass_through} did (wala/ML#513 bucket 2a), so one additional reassignment is counted as a
   * tensor local. The widened operand walks (wala/ML#739) add one more: {@code sequence_lengths -
   * 1}'s operator result passes the operand-tensor-evidence gate (wala/ML#451) through the {@code
   * tf.cast} chain and types the runtime-true {@code (2,)} int32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfForward()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_forward",
        4,
        17,
        Map.of(
            2, Set.of(TENSOR_2_2_4_FLOAT32),
            3, Set.of(TENSOR_2_4_FLOAT32),
            4, Set.of(TENSOR_4_4_FLOAT32),
            5, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Pins {@code crf_decode_forward(inputs, state, transition_params, sequence_lengths)}'s parameter
   * types. Function body mirrors {@code crf_decode_forward} from {@code
   * kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py} for tensor-type inference coverage. The caller passes
   * {@code inputs} from {@code x[:, 1:, :]} and {@code state} from {@code x[:, 0, :]}, recovered as
   * {@code (2, 2, 4)} and {@code (2, 4)} respectively via the multi-dim subscript-slice modeling
   * (wala/ML#406): {@code inputs[:, 1:, :]} drops the leading element of the middle axis ({@code 3
   * → 2}) and {@code inputs[:, 0, :]} drops the middle axis entirely (an integer index).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfDecodeForward()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_decode_forward",
        4,
        6,
        Map.of(
            2, Set.of(TENSOR_2_2_4_FLOAT32),
            3, Set.of(TENSOR_2_4_FLOAT32),
            4, Set.of(TENSOR_4_4_FLOAT32),
            5, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Pins {@code _gather_elements_along_row(data, column_indices)}'s parameter types. Function body
   * mirrors {@code _gather_elements_along_row} from {@code
   * deep_recommenders/keras/models/retrieval/sbcnm.py} (identical to {@code _take_long_axis} in
   * {@code factorized_top_k} per the source), a real-world recommender-systems utility, for
   * tensor-type inference coverage. Both parameters infer concretely: {@code data} as {@code (2, 4)
   * float32} and {@code column_indices} as {@code (2, 3) int32}. (The function's
   * runtime-dimensioned final {@code tf.reshape} leaves the local <em>result</em> symbolic, but the
   * parameters themselves are exact.)
   *
   * <p>The local tensor-variable count is {@code 10}. The {@code tf.shape} shape-vector arm
   * (wala/ML#722) types the final reshape to its exact runtime {@code (2, 3)}, and the {@code
   * tf.range} remodel (wala/ML#723) types the {@code range}/{@code expand_dims} chain soundly
   * ({@code (Unresolved,)} and {@code (Unresolved, 1)}); the tile and the flat-index arithmetic
   * stay at sound unknowns, since a fold over an {@code Unresolved} axis has no numeric value.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGatherElementsAlongRow()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gather_elements_along_row.py",
        "_gather_elements_along_row",
        2,
        10,
        Map.of(
            2, Set.of(TENSOR_2_4_FLOAT32),
            3, Set.of(TENSOR_2_3_INT32)));
  }

  /**
   * Pins {@code create_attention_mask_from_input_mask(from_tensor, to_mask)}'s parameter types.
   * Function body mirrors {@code create_attention_mask_from_input_mask} from {@code
   * kyzhouhzau/NLPGNN/nlpgnn/tools.py}, a real-world function that builds a 3D attention mask from
   * a 2D input mask, for tensor-type inference coverage. Both parameters infer concretely.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCreateAttentionMaskFromInputMask()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_create_attention_mask.py",
        "create_attention_mask_from_input_mask",
        2,
        6,
        Map.of(
            2, Set.of(TENSOR_2_3_4_FLOAT32),
            3, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Pins {@code accuracy(y_pred, y_true)}'s parameter types. Function body mirrors {@code accuracy}
   * from {@code YunYang1994/TensorFlow2.0-Examples/2-Basical_Models/Multilayer_Perceptron.py}.
   * Distinct from {@link #testNeuralNetwork4}'s {@code accuracy} (which is the {@code
   * Dense}-layer-chain variant from a different repo); this is the raw-{@code tf.matmul} MLP
   * companion, paired with {@link #testMultilayerPerceptron}.
   *
   * <p>Empirically, both parameters are concrete: {@code y_pred} (vn=2) is {@code (2, 2) float32}
   * and {@code y_true} (vn=3) is {@code (2,) int64}, matching the caller-side {@code tf.constant}
   * shapes.
   */
  @Test
  public void testMlpAccuracy()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_mlp_accuracy.py",
        "accuracy",
        2,
        7,
        Map.of(2, Set.of(TENSOR_2_2_FLOAT32), 3, Set.of(TENSOR_2_INT64)));
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
   * flatten}, {@code d1}, {@code d2}); after the fix for wala/ML#358 (chained {@code Dense} shape
   * propagation), {@code d1} and {@code d2} are now tracked through the SSA-chain fallback. Counts
   * are source-level &mdash; one per distinct value number, deduplicated across the two calling
   * contexts ({@code train_step}/{@code test_step}; wala/ML#371, Option 2) &mdash; so the count is
   * 4 (parameter plus three registered intermediates) rather than the context-multiplied 8. The
   * residual gap from 4 to the rule-based 5 is one intermediate that still doesn't register; see
   * wala/ML#389.
   *
   * <p>With the count check passing, the test now fails on value 3's type: actual {@code {(32, 28)
   * float32, (16, 28) float32, (28, 28) float32, ? unknown}} &mdash; a union that contains an
   * over-peeled shape ({@code (32, 28)} / {@code (16, 28)} = batch applied to a peeled {@code
   * (28,)}), the unbatched source shape {@code (28, 28)}, and a ⊤ entry. The {@code float32} dtype
   * on three of the four entries indicates that per-index dtype dispatch routes {@code x_train}'s
   * dtype to slot 0 correctly; what is missing is (a) the shape contribution from the {@code
   * x_train[..., tf.newaxis]} subscript that would add the trailing {@code 1} dim, and (b)
   * suppression of the erroneous over-peel path. The earlier labels-swap symptom (shape {@code
   * (32,)} uint8) described under wala/ML#396 appears to have been resolved by the per-index dtype
   * delegation work landed on this branch; the remaining shape mismatch is shape-only.
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
   * A negative k-CFA depth is invalid (the depth is the call-string length for the targeted context
   * selector) and is rejected at construction (wala/ML#379).
   */
  @Test(expected = IllegalArgumentException.class)
  public void testNegativeTargetedCfaDepthRejected() {
    new PythonTensorAnalysisEngine(
        emptyList(), PythonTensorAnalysisEngine.TENSORFLOW, /* targetedCfaDepth= */ -1);
  }

  /**
   * {@code encoder(x)} receives {@code x} from call sites {@code decoder(encoder(x))} inside {@code
   * run_optimization} (training loop) and {@code decoder(encoder(batch_x))} at the module-level
   * test loop. Both call sites pass batches of shape {@code (256, 784)} dtype {@code float32}
   * (verified by Python assert statements in {@code autoencoder.py}).
   *
   * <p>Expected tensor variable count: 11 &mdash; the distinct SSA vns in {@code encoder}'s body
   * that get tensor types: the parameter {@code x} (vn=2) plus the full layer-1 and layer-2
   * computation: {@code weights['encoder_h1']} (vn=19), {@code biases['encoder_b1']} (vn=24),
   * layer-1 matmul (vn=15), layer-1 add (vn=11), {@code layer_1} sigmoid (vn=4), and the
   * corresponding layer-2 values {@code weights['encoder_h2']} (vn=41), {@code
   * biases['encoder_b2']} (vn=45), layer-2 matmul (vn=38), layer-2 add (vn=35), {@code layer_2}
   * sigmoid (vn=31, returned). Counts are source-level &mdash; one per distinct vn, deduplicated
   * across the 2 call-site contexts (wala/ML#371, Option 2) &mdash; so the two contexts no longer
   * double the total to 22.
   */
  @Test
  public void testAutoencoder()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("autoencoder.py", "encoder", 1, 11, Map.of(2, Set.of(TENSOR_256_784_FLOAT32)));
  }

  /**
   * {@code mean_square(reconstructed, original)} is called only from {@code run_optimization}
   * (itself a FUT &mdash; {@link #testAutoencoder3()}). Its arguments are {@code
   * reconstructed_image = decoder(encoder(x))} and {@code x}, both of which have runtime shape
   * {@code (256, 784)} dtype {@code float32}. Direct call-site asserts aren't possible (they would
   * perturb {@code run_optimization}'s count), so the runtime types are verified indirectly through
   * the {@code batch_x} asserts at the training-loop call of {@code run_optimization}.
   *
   * <p>Expected tensor variable count: 4 — 2 parameters plus 2 intermediate-op tensors picked up by
   * #196's {@code ReadDataFallback}: {@code original - reconstructed} (vn=9) and {@code tf.pow}
   * (vn=13), both flow-through {@code (256, 784) float32}. {@code tf.reduce_mean}'s scalar result
   * is the function's return and isn't tracked as a separate variable. Per-op generators tracked in
   * #449 would tighten the asserted types from ⊤-shape/UNKNOWN-dtype to concrete shapes without
   * changing the count.
   *
   * <p>Value 2 ({@code reconstructed}) resolves to concrete {@code (256, 784) float32} after the
   * {@code tensorflow/python/ops/variables/Variable} allocatable-class declaration was added in
   * {@code tensorflow.xml} (closes <a
   * href="https://github.com/wala/WALA/issues/1889">wala/WALA#1889</a>). With the variable
   * allocations now registered in the heap model, the {@code matmul → add → sigmoid →
   * user-function-return} shape chain fully resolves end-to-end.
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
   * <p>Expected local tensor variable count: 5, the tensor-producing values {@code encoder(x)}
   * result, {@code decoder(...) = reconstructed_image}, {@code mean_square(...) = loss}, {@code
   * gradients} (a fresh tensor variable since wala/ML#430's {@code Gradient} generator), and the
   * returned {@code loss} alias. The count dropped from 6 to 5 with wala/ML#750: {@code
   * trainable_variables = list(weights.values()) + list(biases.values())} is a Python list
   * concatenation, not a tensor, so it no longer falsely registers as a tensor variable.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testAutoencoder3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("autoencoder.py", "run_optimization", 1, 5, Map.of(2, Set.of(TENSOR_256_784_FLOAT32)));
  }

  /**
   * {@code decoder(x)} is called from the same two sites as {@code encoder} ({@link
   * #testAutoencoder()}), but with {@code x} being the output of {@code encoder}. Since {@code
   * encoder}'s layer 2 has dim {@code num_hidden_2 = 64}, {@code decoder} receives {@code (256, 64)
   * float32} (verified by a Python assert in the test loop).
   *
   * <p>Expected tensor variable count: 11 (parallel to {@link #testAutoencoder()}). Same body
   * structure: 11 distinct SSA vns covering the parameter plus the two-layer {@code
   * weights[...]}/{@code biases[...]} / matmul / add / sigmoid chain, counted source-level (one per
   * distinct vn, deduplicated across the 2 call-site contexts; wala/ML#371, Option 2).
   *
   * <p>Value 2 ({@code decoder}'s {@code x} parameter) resolves to concrete {@code (256, 64)
   * float32} after the {@code tensorflow/python/ops/variables/Variable} allocatable-class fix
   * (closes <a href="https://github.com/wala/WALA/issues/1889">wala/WALA#1889</a>). The {@code
   * encoder → decoder} shape chain now flows end-to-end through the XML-summary return keys.
   */
  @Test
  public void testAutoencoder4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("autoencoder.py", "decoder", 1, 11, Map.of(2, Set.of(TENSOR_256_64_FLOAT32)));
  }

  @Test
  public void testSigmoid()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sigmoid.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  // wala/ML#449 Tier 2: element-wise unary math ops. Each preserves shape and dtype from the
  // input. Same `Sigmoid`-shape pass-through pattern; the receiving function's parameter at
  // vn=2 carries `tf.constant([1.0, 2.0, 3.0])`'s `(3,) float32`.

  @Test
  public void testAbs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_abs.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testAcos()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_acos.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testExp()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_exp.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testExp()} exercising the keyword-argument call site {@code
   * tf.math.exp(x=...)}. {@link com.ibm.wala.cast.python.ml.client.PassThroughUnaryTensorGenerator}
   * routes argument resolution through {@code getArgumentPointsToSet(builder, paramPos, paramName)}
   * which resolves keyword args via {@code paramName}; without a kwarg fixture that branch is
   * dead-on-arrival in the test data.
   */
  @Test
  public void testExpKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_exp_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testTanh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tanh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testRsqrt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_rsqrt.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Companion to {@link #testRsqrt()} exercising the keyword-argument call site. */
  @Test
  public void testRsqrtKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_rsqrt_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Regression guard for wala/ML#449's closing fix: {@code tf.random.truncated_normal(shape)} now
   * dispatches to the {@code TruncatedNormal} generator (via {@code PROPERTY_NAME_GENERATORS}) and
   * resolves to precise {@code (2, 3) float32}. Pre-fix this fell through to {@code
   * ReadDataFallback} and emitted {@code ⊤ shape / UNKNOWN dtype}.
   */
  @Test
  public void testTruncatedNormal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_truncated_normal.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testLogSoftmax()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_log_softmax.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testLogSoftmax()} exercising the keyword-argument call site {@code
   * tf.nn.log_softmax(logits=...)}.
   */
  @Test
  public void testLogSoftmaxKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_log_softmax_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testL2Normalize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_l2_normalize.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testSigmoid2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sigmoid2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.equal} returns a {@code tf.bool}-dtype tensor with the broadcasted
   * shape of its inputs, regardless of input dtype. Exercises the {@link ComparisonOperation}
   * generator (introduced for wala/ML#427) — the dtype must be BOOL even though both operands are
   * float32.
   */
  @Test
  public void testEqual()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_equal.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_BOOL)));
  }

  /**
   * Same as {@link #testEqual} but for {@code tf.not_equal} — verifies the {@link
   * ComparisonOperation} dispatch scales beyond a single op. Establishes the pattern for the
   * remaining comparison ops ({@code tf.less}, {@code tf.less_equal}, {@code tf.greater}, {@code
   * tf.greater_equal}).
   */
  @Test
  public void testNotEqual()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_not_equal.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_BOOL)));
  }

  /**
   * Counterpart of {@link #testEqual} for {@code tf.less}. Verifies the {@link ComparisonOperation}
   * route emits {@code bool} dtype for the four ordering comparisons (wala/ML#427 residual).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLess()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_less.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_BOOL)));
  }

  /**
   * Counterpart of {@link #testLess} for {@code tf.less_equal}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLessEqual()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_less_equal.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_BOOL)));
  }

  /**
   * Counterpart of {@link #testLess} for {@code tf.greater}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testGreater()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_greater.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_BOOL)));
  }

  /**
   * Counterpart of {@link #testLess} for {@code tf.greater_equal}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testGreaterEqual()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_greater_equal.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_BOOL)));
  }

  /**
   * Regression test for wala/ML#435: a recursive Python function whose return value flows back into
   * itself used to drive {@link
   * com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory#getGenerator(
   * com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable,
   * com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder)} into unbounded recursion
   * via the return-value follow-through and the assignment-graph predecessor walk, ending in {@code
   * StackOverflowError}. The cycle guard added in this PR returns {@code null} when a {@code
   * PointsToSetVariable} is re-encountered along the current call chain. With the cycle guard in
   * place, the recursive call's return value still resolves through its base-case branch — the
   * input {@code tf.constant(1)} (a scalar int32 tensor) flows back to {@code f}'s parameter.
   *
   * <p>The Python test deliberately omits the {@code @tf.function} decorator. Empirically, the
   * regression reproduces without it (verified by reverting the cycle guard locally — this test
   * still SOes), and the decorated form would re-trace the recursive call at runtime and hit
   * Python's recursion limit before the assertions could run. The undecorated form lets {@code
   * python3.10} execute the file to completion with the {@code shape}/{@code dtype} assertions on
   * {@code result} exercised.
   */
  @Test
  public void testRecursiveFunction()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_recursive_function.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Regression test for wala/ML#451 (reproducer 1): a recursive Python function whose only call
   * sites are {@code recursive_fn(5)} (a Python int literal) and {@code recursive_fn(n - 1)} (still
   * a Python int) must not classify its parameter as a tensor. There is no {@code tf.constant}, no
   * decorator, and no tensor anywhere in the program — the analysis should report zero tensor
   * parameters and zero function-local tensor variables.
   */
  @Test
  public void testRecursionIntOnly()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_recursion_int_only.py", "recursive_fn", 0, 0);
  }

  /**
   * Regression test for wala/ML#451 (reproducer 2): a recursive function whose external call site
   * passes a real tensor ({@code recursive_fn(tf.constant(5))}). The parameter {@code n} should be
   * classified as a scalar int32 tensor (the {@code tf.constant} flows through the assignment graph
   * from the caller), and {@code n - 1} inside the body is a tensor binop too — the binop
   * operand-tensor gate in {@link TensorGeneratorFactory} dispatches {@code n - 1} to {@link
   * ElementWiseOperation} because at least one operand ({@code n}) has tensor evidence in its PTS.
   *
   * <p>Tensor-variable count breakdown: {@code vn=2} (parameter) is seen across two analysis
   * contexts (the top-level call and the recursive self-call), and {@code vn=10} ({@code n - 1}) is
   * the binop result in the top-level context. Counts are source-level &mdash; one per distinct
   * value number, deduplicated across calling contexts (wala/ML#371, Option 2) &mdash; so the
   * parameter's two contexts collapse and the count is 2 ({@code vn=2} and {@code vn=10}), not the
   * three {@code (CGNode, vn)} entries the analysis registers.
   */
  @Test
  public void testRecursionTensorOnly()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_recursion_tensor_only.py",
        "recursive_fn",
        1,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Regression test for <a href="https://github.com/wala/ML/issues/750">wala/ML#750</a>: a Python
   * list concatenation accumulated in a loop ({@code result_array += [on, off]}) is not a tensor.
   * The pattern is vendored from {@code MusicTransformer-tensorflow2.0}'s {@code
   * midi_processor/processor.py} {@code _divide_note}, where {@code on}/{@code off} are plain
   * {@code SplitNote} objects and {@code result_array} is a list. The concatenation's {@code +}
   * binop lands in {@link
   * com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine#getDataflowSources}'s {@code
   * SSABinaryOpInstruction} branch, so it is gated by operand tensor evidence. The loop-carried
   * {@code result_array} is a phi over the initial {@code []} allocation and the previous
   * concatenation, so its {@code list} allocation is reached only through its points-to set, not
   * its own def. Before the fix, the points-to-set tensor-evidence check counted that {@code list}
   * allocation as evidence, so the binop dispatched to {@code ElementWiseOperation} and typed the
   * result as a tensor; that type then fed back through the loop into {@code result_array}, a
   * self-reinforcing false tensor with a {@code TENSORFLOW} origin. {@code _divide_note} must have
   * no tensor variables.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testListConcatNotTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf750_list_concat.py", "_divide_note", 0, 0);
  }

  /**
   * Companion to {@link #testListConcatNotTensor()} for the {@code tuple} arm of the wala/ML#750
   * fix. {@code _divide_note_tuple} accumulates a loop-carried {@code tuple} with {@code
   * result_tuple += (on, off)}. Like the list case, the {@code +} binop is gated by operand tensor
   * evidence, and the loop-carried {@code tuple}'s allocation is reached only through its points-to
   * set. A {@code tuple} concatenation is not a tensor op, so the accumulator's {@code tuple}
   * allocation must not count as tensor evidence and {@code _divide_note_tuple} must have no tensor
   * variables.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testTupleConcatNotTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf750_list_concat.py", "_divide_note_tuple", 0, 0);
  }

  /**
   * Regression test for <a href="https://github.com/wala/ML/issues/653">wala/ML#653</a>: Python
   * list repetition ({@code [0] * 3}) is not a tensor. The {@code *} binop has a {@code list}
   * operand and an {@code int} operand, so it is list repetition (producing a list), not tensor
   * scalar-multiplication. The binop operand-tensor gate must not treat the bare {@code list}
   * operand as tensor evidence, so {@code consume}'s parameter is not classified as a tensor.
   */
  @Test
  public void testListRepetition()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_list_repetition.py", "consume", 0, 0);
  }

  /**
   * Regression test for wala/ML#451 (reopen): asserts the underlying PA state that Hybridize's
   * {@code Function.inferPrimitiveParameters} consumes &mdash; specifically, that no primitive
   * {@link ConstantKey} is reachable through the parameter's points-to set when traversed
   * transitively through instance fields. The traversal here mirrors the recursion in {@code
   * Function.containsPrimitive(InstanceKey, ...)}: a {@code ConstantKey} with a non-null value
   * (excluding bools) is "primitive"; an {@link AllocationSiteInNode} or {@link ConcreteTypeKey} is
   * recursively examined through its declared instance fields.
   *
   * <p>The fixture is the same as {@link #testRecursionTensorOnly()} ({@code recursive_fn(tf
   * .constant(5))}). Pre-fix, this assertion failed because {@code tensorflow.xml}'s {@code
   * tf.constant.do} method bound the user's {@code value} argument to the alloc's {@code value}
   * field via {@code <putfield>}, so the field-traversal walk found the user's {@code
   * ConstantKey<Integer:5>} and classified the parameter as primitive even though the alloc IS a
   * tensor producer. The XML now omits that binding (the {@link
   * com.ibm.wala.cast.python.ml.client.Constant} generator reads dtype/shape directly from the
   * call's value-arg PTS rather than from the alloc's field), and a CG-walk fallback in {@code
   * TensorGenerator.getShapesFromShapeArgument} keeps shape inference working for cases like {@code
   * tf.constant([2, 3])} as a shape argument.
   */
  @Test
  public void testRecursionTensorOnlyHasNoPrimitiveInPTS()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(emptyList(), new String[] {"tf2_test_recursion_tensor_only.py"});
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);
    engine.performAnalysis(builder);

    String functionSignature = "script tf2_test_recursion_tensor_only.py.recursive_fn.do()LRoot;";
    boolean checkedAtLeastOneContext = false;

    for (CGNode node : CG) {
      if (!node.getMethod().getSignature().equals(functionSignature)) continue;
      // Parameter `n` is at vn=2 (vn=1 is `self`/function object).
      PointerKey paramPK =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
      OrdinalSet<InstanceKey> pts = builder.getPointerAnalysis().getPointsToSet(paramPK);
      if (pts == null || pts.isEmpty()) continue;
      checkedAtLeastOneContext = true;
      for (InstanceKey ik : pts) {
        Set<InstanceKey> seen = new HashSet<>();
        assertTrue(
            "Parameter `n` of recursive_fn(tf.constant(5)) should not have any primitive"
                + " ConstantKey reachable through PA field traversal in context "
                + node.getContext()
                + " (instance="
                + ik
                + "). This is the underlying state that Hybridize's"
                + " Function.containsPrimitive consumes (wala/ML#451 reopen).",
            !containsPrimitiveByFieldWalk(ik, builder.getPointerAnalysis(), seen));
      }
    }
    assertTrue(
        "Expected to check at least one CGNode/context for recursive_fn with non-empty PTS"
            + " for vn=2; if this assertion fails, the test setup may not have produced any"
            + " analyzable parameter.",
        checkedAtLeastOneContext);
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>: a custom
   * {@code fit} iterates an {@code experimental_distribute_dataset}-wrapped {@code tf.data} dataset
   * and threads the yielded {@code (inputs, targets)} tuple into {@code train_step}. The strategy's
   * {@code experimental_distribute_dataset} is a pass-through of its dataset argument, so the
   * distributed dataset stays a recognized tensor iterable and {@code train_step}'s {@code inputs}
   * and {@code targets} parameters type to {@code (2,)} float32 rather than being dropped (which
   * cascaded to every downstream consumer).
   */
  @Test
  public void testDistributeFitTupleParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_distribute_fit_tuple_param.py",
        "Model.train_step",
        2,
        3,
        Map.of(3, Set.of(TensorType.of(FLOAT_32, 2)), 4, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/623">wala/ML#623</a>: a {@code
   * padded_batch} element threaded through a custom {@code fit} into {@code train_step} types the
   * parameters. {@code padded_batch} was unmodeled, so the per-element tensor type was dropped
   * before reaching the callee; modeling it like {@code batch} recovers it. The two parameters type
   * to {@code (2, 2)} int32 (the batch dimension prepended to the {@code (2,)} element).
   */
  @Test
  public void testPaddedBatchParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = TensorType.of(INT_32, 2, 2);

    test(
        "tf2_test_padded_batch_param.py",
        "Model.train_step",
        2,
        3,
        Map.of(3, Set.of(t), 4, Set.of(t)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>'s gpt-2
   * case: a callee parameter that receives a tensor argument at a direct method-call site is typed
   * by Ariadne. The {@code Gpt2} model, dataset pipeline, and {@code _train_step}/{@code
   * train_step}/{@code get_loss} dispatch are vendored verbatim from {@code
   * akanyaani/gpt-2-tensorflow2.0} ({@code gpt2_model.py}, {@code data_pipeline.py}); only the
   * transformer {@code layers} (and the {@code utils}/{@code scripts} helpers) are stubbed to
   * pass-throughs, since {@code get_loss}'s parameter typing does not depend on the model body.
   *
   * <p>A {@code padded_batch} dataset element flows through {@code fit} to the {@code
   * @tf.function(input_signature=...)}-decorated {@code train_step}, then {@code _train_step}, then
   * {@code get_loss(targets, predictions)}. The {@code real} parameter (vn=3), bound to the
   * dataset-sourced {@code targets}, types to {@code (2, 2)} int32, so Ariadne emits the parameter
   * type for this exact shape. With wala/ML#665 forwarding wildcard import bindings, {@code
   * pred} types too: the stubbed model body's forward output is a rank-3 union with the vocab
   * dimension recovered. This pins that wala/ML#618's residual gpt-2 failure is downstream of
   * Ariadne, not an emission gap.
   */
  @Test
  public void testGpt2InterprocGetLoss()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_proj/layers/__init__.py", "gpt2_proj/layers/embedding_layer.py",
          "gpt2_proj/layers/feed_forward.py", "gpt2_proj/layers/layer_norm.py",
          "gpt2_proj/layers/attention_layer.py", "gpt2_proj/utils/__init__.py",
          "gpt2_proj/utils/tf_utils.py", "gpt2_proj/scripts/__init__.py",
          "gpt2_proj/scripts/utils.py", "gpt2_proj/data_pipeline.py",
          "gpt2_proj/gpt2_model.py", "gpt2_proj/tf2_test_gpt2_probe.py"
        },
        "gpt2_model.py",
        "Gpt2.get_loss",
        "gpt2_proj",
        2,
        8,
        Map.of(
            3,
            Set.of(TensorType.of(INT_32, 2, 2)),
            4,
            // The model forward output: rank 3 with the vocab dimension recovered as the
            // constant 100; the dtype is refinable once `add_weight` consumes its `dtype`
            // argument. The (2, 2) int32 member is the call-site union with the label tensor.
            Set.of(
                new TensorType(
                    UNKNOWN,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, new NumericDim(100))),
                TensorType.of(INT_32, 2, 2))));
  }

  /**
   * The full subject from {@code akanyaani/gpt-2-tensorflow2.0} (a perf-eval corpus subject)
   * vendored verbatim with the REAL transformer layers and {@code input_fn}, unlike the stubbed
   * {@link #testGpt2InterprocGetLoss()}. {@code get_loss}'s {@code real} now types end to end
   * (wala/ML#618), resolving the whole dataset-sourced chain that previously dehybridized it.
   *
   * <p>The chain: {@code real} (the dataset-sourced {@code targets}) is built in {@code input_fn}
   * as {@code
   * tf.data.TFRecordDataset(...).map(parse_example).padded_batch(...).repeat(...).prefetch(...)},
   * passed as a list {@code _model.fit([_train, _test], ...)}, list-unpacked, iterated with {@code
   * enumerate} and nested unpacking, and forwarded through an indirected {@code train_fuc} into
   * {@code train_step} → {@code _train_step} → {@code get_loss}. Each layer is modeled: {@code
   * parse_example} densifies a {@code tf.io.VarLenFeature} through a dict to {@code (?,)} int32
   * (wala/ML#646, pinned by {@link #testParseExampleTuple()}); {@code map} types the element from
   * {@code map_func}'s return (wala/ML#506, {@link #testDatasetMapTuple()}); a pass-through
   * transform after {@code map} keeps the mapped type (wala/ML#649, {@link
   * #testDatasetMapRepeat()}); {@code TFRecordDataset} is chainable ({@link #testTfrecordMap()});
   * the dataset survives the list (wala/ML#648, {@link #testFitLoop()}); and the {@code
   * padded_batch} dims apply (wala/ML#673). So {@code real} resolves to the batched element {@code
   * (32, ?)} int32 (the pipeline's {@code batch_size=32} with the pad-to-longest sequence dim),
   * unioned with the standard partial-batch sibling {@code (?, ?)}.
   *
   * <p>{@code pred} types too (wala/ML#665): the model forward output is a tensor union. With
   * {@code add_weight} consuming its {@code shape}/{@code dtype} arguments (wala/ML#667),
   * constructor keyword arguments forwarded to {@code __init__} (wala/ML#664), the wala/ML#739
   * operand-walk repairs, and parameter defaults materializing in the pointer analysis
   * (wala/ML#743), the decoder stack resolves end to end under the fit-path contexts: the
   * runtime-true logits form is {@code (Dynamic, Dynamic, 10)} float32 (the batch and sequence axes
   * ride the dataset's dynamic dims), alongside the {@code (?, ?, 10)} partial. The wala/ML#680
   * {@code unknown}-dtype phantom is gone: with the decoder-stack output resolving, {@code
   * OutputLayer.call}'s dead {@code self.porj_weights} arm no longer contributes a member. The
   * union is the order-independent fixed point (wala/ML#674): identical across runs and across
   * suite/single-test modes. Analyzed statically here, like the consumer's vendoring; it runs in
   * the perf-eval with its tfrecord/data setup.
   *
   * <p>TODO: Drop the {@code (Dynamic, Dynamic, 8, 8)} member once <a
   * href="https://github.com/wala/ML/issues/746">wala/ML#746</a> filters constant-decidable branch
   * arms per call site; it is the {@code mode="projection"} call's rank-3 input crossing into the
   * embedding-mode lookup, runtime-infeasible at that site.
   */
  @Test
  public void testGpt2GetLossVendored()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_vendored/layers/__init__.py", "gpt2_vendored/layers/embedding_layer.py",
          "gpt2_vendored/layers/feed_forward.py", "gpt2_vendored/layers/layer_norm.py",
          "gpt2_vendored/layers/attention_layer.py", "gpt2_vendored/utils/__init__.py",
          "gpt2_vendored/utils/tf_utils.py", "gpt2_vendored/scripts/__init__.py",
          "gpt2_vendored/scripts/utils.py", "gpt2_vendored/data_pipeline.py",
          "gpt2_vendored/A.py"
        },
        "A.py",
        "Gpt2.get_loss",
        "gpt2_vendored",
        2,
        8,
        Map.of(
            3,
            Set.of(new TensorType(INT_32, asList(DynamicDim.INSTANCE))),
            4,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, new NumericDim(10))),
                new TensorType(
                    FLOAT_32, asList(DynamicDim.INSTANCE, DynamicDim.INSTANCE, new NumericDim(10))),
                new TensorType(
                    FLOAT_32,
                    asList(
                        DynamicDim.INSTANCE,
                        DynamicDim.INSTANCE,
                        new NumericDim(8),
                        new NumericDim(8))))));
  }

  /**
   * Regression guard for wala/ML#645: {@code tf.io.VarLenFeature(dtype)} models the SparseTensor a
   * variable-length feature parses to, so {@code tf.sparse.to_dense} of it types from the feature's
   * dtype ({@code int64}) and the API-contract shape (rank-1 with a dynamic length, {@code (?,)}).
   * The {@code io}-registration fix makes {@code tf.io.*} resolve at all (they were registered
   * under {@code tf}, not the {@code io} object). The rank-1 dynamic shape is the contract-model
   * refinement of wala/ML#647 (formerly ⊤).
   */
  @Test
  public void testVarLenFeature()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_var_len_feature.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_DYNAMIC_INT64)));
  }

  /**
   * The realistic gpt-2 shape for wala/ML#645: a {@code VarLenFeature} in a feature dict, parsed by
   * {@code tf.io.parse_single_example}, subscripted, and densified. {@code consume}'s parameter
   * types to {@code (?,)} int64: the VarLenFeature SparseTensor now keeps a live points-to set
   * through the dict {@code putfield}/{@code getfield} (wala/ML#646), and {@code
   * tf.sparse.to_dense} resolves the dict-routed operand by dispatching the {@code VarLenFeature}
   * generator at the SparseTensor's allocation site, recovering the feature's dtype and the rank-1
   * dynamic (contract) shape. The static shape is {@code (?,)}, not the concrete {@code (2,)} the
   * Python runtime produces, because the example's length is lost across the serialize/parse
   * round-trip. This is the dict-routed companion to the direct {@link #testVarLenFeature()};
   * together they un-strand {@code get_loss}'s {@code real} in {@link #testGpt2GetLossVendored()}.
   */
  @Test
  public void testParseSingleExample()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_parse_single_example.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_DYNAMIC_INT64)));
  }

  /**
   * Mirrors gpt-2's {@code parse_example} for wala/ML#618: a tuple return over two {@code
   * tf.cast(tf.sparse.to_dense(parsed[k]), tf.int32)} values, each parsed from a {@code
   * VarLenFeature} in a feature dict, but called directly (no {@code dataset.map}). The recovered
   * {@code (?,)} int64 propagates through {@code to_dense}, the int32 cast, the tuple return, and
   * the destructuring, so {@code consume}'s parameter ({@code targets}) types to {@code (?,)}
   * int32. Together with {@link #testParseSingleExample()} this isolates the {@code dataset.map}
   * element-type layer as the sole remaining gap for the full {@link #testGpt2GetLossVendored()}
   * subject.
   */
  @Test
  public void testParseExampleTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_parse_example_tuple.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_32, asList(DynamicDim.INSTANCE)))));
  }

  /**
   * The gpt-2 {@code fit}-side shape for wala/ML#618, end to end: a mapped dataset passed in a list
   * ({@code fit([ds, ds])}), list-unpacked, iterated with {@code enumerate} and nested unpacking,
   * then forwarded through an indirected callback into {@code get_loss}. {@code real} (the second
   * mapped tuple component) types to {@code (4,)} int64, exercising wala/ML#648 (dataset through a
   * list) together with wala/ML#506 (the {@code map} tuple element). The full vendored subject
   * ({@link #testGpt2GetLossVendored()}) additionally chains {@code padded_batch}/{@code repeat}/
   * {@code prefetch} behind an {@code input_fn} return, which still loses the transformation chain.
   */
  @Test
  public void testFitLoop()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_fit_loop.py", "consume", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 4))));
  }

  /**
   * Regression guard for wala/ML#618: {@code tf.data.TFRecordDataset(...)} is a chainable dataset,
   * so {@code .map(parse_example)} resolves and the VarLenFeature-parsed {@code targets} types to
   * {@code (?,)} int32. Previously {@code TFRecordDataset} was a bare {@code Dataset} field with no
   * {@code do()}, so the chain did not resolve.
   */
  @Test
  public void testTfrecordMap()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tfrecord_map.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_32, asList(DynamicDim.INSTANCE)))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>: {@code
   * strategy.run(fn, (a, b))} forwards both elements of the positional {@code args} tuple into the
   * two-parameter callback {@code step_fn(inp, tar)}, not just the first. Both {@code
   * consume_inp}'s and {@code consume_tar}'s parameters type to {@code (2,)} int32; previously the
   * {@code tensorflow/distribute/run/run} model forwarded only {@code args[0]}, so {@code tar} (and
   * {@code consume_tar}'s parameter) stayed untyped.
   */
  @Test
  public void testStrategyRunTwoArgsInp()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_strategy_run_two_args.py",
        "consume_inp",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 2))));
  }

  /**
   * Companion to {@link #testStrategyRunTwoArgsInp()} pinning the <em>second</em> forwarded
   * element: {@code consume_tar}'s parameter types to {@code (2,)} int32, confirming {@code
   * strategy.run} forwards {@code args[1]}. See <a
   * href="https://github.com/wala/ML/issues/618">wala/ML#618</a>.
   */
  @Test
  public void testStrategyRunTwoArgsTar()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_strategy_run_two_args.py",
        "consume_tar",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 2))));
  }

  /**
   * Companion to {@link #testGpt2InterprocGetLoss()} that drives the <em>distributed</em> reach of
   * <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>'s gpt-2 case: the same vendored
   * {@code Gpt2} model, but reached via {@code distributed_train_step} &rarr; {@code
   * _distributed_train_step} &rarr; {@code mirrored_strategy.run(step_fn, args=(inputs, targets))}
   * &rarr; {@code step_fn} &rarr; {@code get_loss(tar, logits)}, the path the real subject takes.
   *
   * <p>{@code get_loss}'s {@code real} parameter (vn=3), bound to the dataset-sourced {@code
   * targets} that flows through {@code strategy.run}'s {@code args} tuple into {@code step_fn}'s
   * {@code tar}, types to {@code (2, 2)} int32 exactly as in the direct reach. This exercises both
   * halves of the wala/ML#618 distributed-reach fix: the {@code tensorflow/distribute/run/run}
   * model forwarding both tuple elements (see {@link #testStrategyRunTwoArgsInp()}), and the {@code
   * args} parameter name surviving summary loading (the <a
   * href="https://github.com/wala/WALA/pull/1972">wala/WALA#1972</a> fix to {@code
   * XMLMethodSummaryReader}'s name filter), without which the keyword {@code args=} could not bind.
   */
  @Test
  public void testGpt2DistributedGetLoss()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_proj/layers/__init__.py", "gpt2_proj/layers/embedding_layer.py",
          "gpt2_proj/layers/feed_forward.py", "gpt2_proj/layers/layer_norm.py",
          "gpt2_proj/layers/attention_layer.py", "gpt2_proj/utils/__init__.py",
          "gpt2_proj/utils/tf_utils.py", "gpt2_proj/scripts/__init__.py",
          "gpt2_proj/scripts/utils.py", "gpt2_proj/data_pipeline.py",
          "gpt2_proj/gpt2_model.py", "gpt2_proj/tf2_test_gpt2_distributed_probe.py"
        },
        "gpt2_model.py",
        "Gpt2.get_loss",
        "gpt2_proj",
        2,
        8,
        Map.of(
            3,
            Set.of(TensorType.of(INT_32, 2, 2)),
            4,
            // The model forward output: rank 3 with the vocab dimension recovered as the
            // constant 100; the dtype is refinable once `add_weight` consumes its `dtype`
            // argument. The (2, 2) int32 member is the call-site union with the label tensor.
            Set.of(
                new TensorType(
                    UNKNOWN,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, new NumericDim(100))),
                TensorType.of(INT_32, 2, 2))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>: a tensor
   * passed interprocedurally to a callee types the callee's parameter. {@code Model.get_loss}'s
   * {@code real} and {@code pred} receive {@code tf.constant} tensors via {@code train_step}, so
   * both type to {@code (3,)} float32 rather than being missed. With the widened operand walks
   * (wala/ML#739), all three body producers count as tensor locals with their runtime-true shapes:
   * {@code pred - real} and {@code tf.square} at {@code (3,)} and the {@code tf.reduce_mean} result
   * at scalar rank 0.
   */
  @Test
  public void testInterprocTensorParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = TensorType.of(FLOAT_32, 3);

    test(
        "tf2_test_interproc_tensor_param.py",
        "Model.get_loss",
        2,
        5,
        Map.of(3, Set.of(t), 4, Set.of(t)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/642">wala/ML#642</a>: a
   * faithful copy of <a href="https://github.com/wala/ML/issues/637">wala/ML#637</a>'s example,
   * {@code tf.constant([1 + 2j, 3 + 4j], dtype=tf.complex64)}. The complex literal ({@code 2j})
   * folds to a {@code PyComplex} whose {@code asInt()} raises a TypeError; before wala/ML#642 that
   * uncaught exception aborted the module's front-end translation and emptied its entrypoint set.
   * The front end now skips folding it, so the constant builds and types {@code consume}'s
   * parameter to complex64. The shape is unknown (⊤) rather than {@code (2,)}: skipping the fold
   * leaves the list elements non-constant, so the size isn't recovered (the integer-valued {@link
   * #testConstantComplex64()} still gets {@code (2,)}). TODO: recover the shape, tracked by <a
   * href="https://github.com/wala/ML/issues/644">wala/ML#644</a>.
   */
  @Test
  public void testComplexLiteralEntrypoint()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_complex_literal.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(new TensorType(COMPLEX64, null))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/640">wala/ML#640</a>: the
   * constant folder evaluates a foldable expression (in an uncalled function) that raises at
   * evaluation time -- here {@code 1 / 0} ({@code ZeroDivisionError}); the original NLPGNN case was
   * a {@code NameError} on a free name. Folding must skip such an eval-time error rather than abort
   * the class hierarchy. If the hierarchy builds, {@code consume}'s parameter types normally to
   * {@code (3,)} int32.
   */
  @Test
  public void testFoldingEvalError()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_folding_eval_error.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>: a tensor
   * passed to a Keras {@code call} method types its parameter. {@code BiLSTM.call}'s {@code inputs}
   * receives a token-id tensor (which then feeds an {@code Embedding}), so it types to {@code (1,
   * 3)} int32 rather than being missed.
   */
  @Test
  public void testKerasCallEmbeddingParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_keras_call_embedding_param.py",
        "BiLSTM.call",
        1,
        2,
        Map.of(3, Set.of(TensorType.of(INT_32, 1, 3))));
  }

  @Test
  public void testStrictnessFailure()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_strictness_failure.py",
        "test_strictness",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_INT32)));
  }

  @Test
  public void testNoneDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_none_dtype.py", "test_none_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
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

  /**
   * {@code run_optimization(x, y)} receives batched CIFAR-10 data (not MNIST despite the file's
   * docstring). At runtime {@code x} has shape {@code (4096, 32, 32, 3)} dtype {@code float32} and
   * {@code y} has shape {@code (4096,)} dtype {@code uint8} (verified by Python assert statements
   * in {@code multigpu_training.py}).
   *
   * <p>Master's types for values 2 and 3 are {@code MNIST_INPUT} &mdash; the MNIST shape {@code (n,
   * 28, 28)} &mdash; which is <em>confidently wrong</em> for CIFAR-10 data (wala/ML#393: {@code
   * keras.datasets.X.load_data()} is seeded uniformly as MNIST-shaped regardless of which dataset
   * module {@code X} is). Branch now hardcodes {@code tf.keras.datasets.cifar10.load_data()} as an
   * intrinsic (paralleling the MNIST modeling), so value 2 correctly reports {@code (4096, 32, 32,
   * 3) float32}. Value 3 (labels) now correctly reports {@code (4096,) uint8} after wala/ML#410
   * landed the top-level {@code np.reshape(x, shape)} modeling: {@code y_train =
   * np.reshape(y_train, (-1))} at line 66 of the source is the function form of reshape (as opposed
   * to the method form {@code x.reshape(...)} which was already handled by {@link NdarrayReshape}).
   * The fix added a {@code numpy/reshape} class to {@code numpy.xml} paired with an {@link
   * NpReshape} generator that reuses {@link Reshape}'s {@code -1}-inference logic and also accepts
   * a bare integer as the shape argument (the parenthesised {@code (-1)} parses as {@code -1}, not
   * a 1-tuple).
   *
   * <p>Expected tensor variable count: 5. The historical trajectory is: pre-wala/ML#451 the count
   * was 5 because of a spurious classification at {@code vn=44} ({@code gpu_batch_size =
   * int(batch_size / num_gpus)} at line 222) where {@code batch_size / num_gpus} is a binop on
   * Python ints, dispatched to {@link ElementWiseOperation} and typed {@code [] of int32} via
   * {@link TensorGenerator#getDTypesOfValue}'s Integer-constant → INT32 path. wala/ML#451's binop
   * operand-tensor gate rejected that classification, dropping the count to the master baseline of
   * 4. wala/ML#430's {@code Gradient} generator then added one fresh tensor per {@code
   * tape.gradient(...)} call (one such call here), bringing the count back to 5 — but now backed by
   * the legitimate registration of the gradient result rather than the spurious binop pickup. See
   * also wala/ML#361 (MNIST modeling) and wala/ML#393 (CIFAR-10 modeling, closed by the commit that
   * landed this test's partial pass).
   */
  @Test
  public void testMultiGPUTraining()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "multigpu_training.py",
        "run_optimization",
        2,
        5,
        Map.of(2, Set.of(TENSOR_4096_32_32_3_FLOAT32), 3, Set.of(TENSOR_4096_UINT8)));
  }

  /**
   * Companion to {@link #testMultiGPUTraining()} on the {@code average_gradients} function in the
   * same fixture. Exercises tensor classification through a per-tower gradient-averaging loop: for
   * each gradient {@code g} in the input tower-gradient list, the body computes {@code
   * tf.expand_dims(g, 0)}, collects the results into a list, and reduces them with {@code
   * tf.concat(axis=0, values=grads)}. Verifies the analyzer detects the 3 internal tensor variables
   * this loop produces: {@code tf.expand_dims(g, 0)}'s dedicated {@code <new>} allocation, the
   * post-allocation receiver, and {@code tf.concat(...)} flowing through <a
   * href="https://github.com/wala/ML/issues/196">wala/ML#196</a>'s {@link
   * com.ibm.wala.cast.python.ml.client.ReadDataFallback}.
   *
   * <p>With `list.append` modeled (<a
   * href="https://github.com/wala/ML/issues/136">wala/ML#136</a>), the loop's element values reach
   * classification and the local-tensor count is 8 (the loop variable {@code g}, the per-iteration
   * {@code expand_dims} results, and the {@code concat} chain). The parameter count stays 0: {@code
   * tower_grads} is the list <em>container</em>, which is not itself a tensor; its elements are
   * classified where they are read.
   */
  @Test
  public void testMultiGPUTraining2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("multigpu_training.py", "average_gradients", 0, 8);
  }

  // wala/ML#449 Tier 3: reductions. Each collapses dims along `axis` (default `None` = all dims)
  // and preserves dtype from input (except `reduce_all` which is always BOOL).

  /**
   * Verifies that {@code tf.estimator.EstimatorSpec(...)} produces a fresh allocation with each
   * named parameter stored as a field on the result. The test reads {@code spec.loss} and asserts
   * that it round-trips back to the original {@code loss_tensor} (a scalar float32). If
   * EstimatorSpec is mis-modeled as "return one of the inputs" instead of "fresh allocation with
   * field sets," the read would either return the wrong dtype or fail to resolve. Exercises the
   * binding + body fix from wala/ML#429.
   */
  @Test
  public void testEstimatorSpec()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_estimator_spec.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Regression test for <a href="https://github.com/wala/ML/issues/523">wala/ML#523</a>: the {@code
   * EstimatorSpec.do()} constructor's fresh allocation must be a {@code
   * Ltensorflow/estimator/EstimatorSpec} (a namedtuple-like spec object) rather than a {@code
   * Ltensorflow/python/framework/ops/Tensor}. The fixture passes the {@code spec} directly to
   * {@code f}; if {@code spec} is misclassified as a tensor allocation, {@code f}'s parameter would
   * be a tensor (and the expected count would be 1 / 1). With the correct non-tensor class, {@code
   * f} has 0 tensor parameters and 0 tensor variables.
   */
  @Test
  public void testEstimatorSpecNotTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_estimator_spec_not_tensor.py", "f", 0, 0, Map.of());
  }

  // wala/ML#449 Tier 4: intrinsic-API ops with fixed output shape and dtype. Both `tf.rank` and
  // `tf.size` return scalar int32 regardless of input. Same hardcoded-output shape as
  // `DatasetRangeGenerator`'s `[] of int64`.

  @Test
  public void testRank()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_rank.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testSize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_size.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  // wala/ML#449 Tier 6: argmax/argmin produce int64 indices. Output shape is the input shape with
  // the `axis` dimension removed (via `ReduceMean`), unblocked by the per-context layer-output
  // fix (wala/ML#530) that stopped `testNeuralNetwork*` from regressing on the
  // `ElementWiseOperation`
  // cross-context cartesian pair.

  // wala/ML#449 Tier 7: linspace/broadcast_to. Shape derives from a shape-arg (`num`/`shape`),
  // dtype derives from a value-arg (`start`/`input`).

  /**
   * Verifies that {@code tf.linspace(0.0, 10.0, 5)} routes through the dedicated {@link Linspace}
   * generator and emits the precise rank-1 shape {@code (5,)} with {@code float32} dtype (derived
   * from the float-typed {@code start} argument).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLinspace()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_linspace.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  /**
   * Companion to {@link #testLinspace()} that exercises the integer-promotion branch in {@link
   * com.ibm.wala.cast.python.ml.client.Linspace#getDefaultDTypes}. {@code tf.linspace} with integer
   * {@code start}/{@code stop} promotes the output to {@code float64} (verified on TF 2.9), not
   * {@code float32}. The float-input case is covered by {@link #testLinspace()}.
   */
  @Test
  public void testLinspaceInt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_linspace_int.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT64)));
  }

  /**
   * Drives the {@code axis}-passed branch in {@link
   * com.ibm.wala.cast.python.ml.client.Linspace#getDefaultShapes}. With {@code axis=1} and vector
   * {@code start}/{@code stop}, {@code tf.linspace} interpolates along axis 1, producing a
   * higher-rank result whose runtime shape is {@code (2, 5)} with dtype {@code float32}.
   *
   * <p>The static analysis currently returns ⊤ (unknown shape) for any axis-passed call — combining
   * {@code start}'s rank with {@code num} is not yet implemented; the generator trades precision
   * for soundness. The assertion encodes the observed result with a TODO pointing at the precision
   * improvement that would narrow it.
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/475">wala/ML#475</a> is fixed and
   * {@code Linspace.getDefaultShapes} computes the precise output shape from {@code
   * start.shape[:axis] + (num,) + start.shape[axis:]}, narrow the assertion to {@code
   * Set.of(TENSOR_2_5_FLOAT32)} (a new constant).
   *
   * <p>Companion to {@link #testLinspace()} (covering the absent-axis branch).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLinspaceAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_linspace_axis.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.broadcast_to(x, [2, 3])} routes through the dedicated {@link
   * BroadcastTo} generator and emits shape {@code (2, 3)} (read from the {@code shape} argument's
   * literal list) with {@code float32} dtype (derived from the {@code input} tensor).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBroadcastTo()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_broadcast_to.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Regression guard for {@code tf.broadcast_to(x, tf.shape(y))} &mdash; the runtime-tensor
   * shape-arg pattern. {@link com.ibm.wala.cast.python.ml.client.TensorGenerator
   * #getShapesFromShapeArgument} throws {@link IllegalStateException} for the runtime {@code
   * Ltensorflow/python/framework/ops/Tensor} that {@code tf.shape(y)} now allocates (post the
   * wala/ML#489 root-cause fix on this PR's `tensorflow.xml`); {@link
   * com.ibm.wala.cast.python.ml.client.BroadcastTo#getDefaultShapes}'s try/catch returns {@code
   * null} (lattice ⊤) instead of letting the exception abort the analysis. The result is shape ⊤
   * with dtype inherited from {@code x} (float32). Without the catch, analysis aborts and this test
   * fails &mdash; this is the direct regression guard for this PR's localized-tolerance fix.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/473">wala/ML#473</a>): the runtime answer is
   * (2, 3) of float32. When the helper learns to recognize {@code tf.shape(y)} as a shape arg
   * (rather than treating it as an unmodeled runtime tensor), tighten this assertion from {@link
   * #TENSOR_UNKNOWN_SHAPE_FLOAT32} to {@code TENSOR_2_3_FLOAT32}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBroadcastToRuntimeShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_broadcast_to_runtime_shape.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.linalg.tensordot} with a scalar {@code axes}. Output
   * dtype is inherited from the {@code a} input (here float32); with {@code axes=1} the output
   * shape is {@code a.shape[:-1] + b.shape[1:]}, so two (2, 2) inputs yield (2, 2). See {@link
   * com.ibm.wala.cast.python.ml.client.Tensordot} (wala/ML#449).
   */
  @Test
  public void testTensordot()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensordot.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.linalg.trace}. Output dtype is inherited from the {@code
   * x} input (here float32); the output shape is the input shape with the last two dimensions
   * dropped, so a (2, 2) input yields a scalar. See {@link
   * com.ibm.wala.cast.python.ml.client.Trace} (wala/ML#449).
   */
  @Test
  public void testTrace()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_trace.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.linalg.trace} on a batched input. The trace collapses the
   * last two dimensions, so a (3, 2, 2) input yields a (3,) output that inherits the input dtype.
   * Exercises the leading-dimensions path of {@link com.ibm.wala.cast.python.ml.client.Trace}
   * (wala/ML#449).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTraceBatched()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_trace_batched.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_3_2_2_FLOAT32),
            3, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.transpose(a)} with no {@code perm} reverses the axes, so a {@code (2,
   * 3)} input yields {@code (3, 2)}. Previously modeled as a first-argument {@code pass_through},
   * which reported the input shape unchanged. See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTransposeDefault()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_transpose.py",
        "consume_transpose_default",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.transpose(a, perm)} with a constant {@code perm} permutes the axes so
   * output axis {@code i} is input axis {@code perm[i]}: a {@code (2, 3, 4)} input with {@code perm
   * = [0, 2, 1]} yields {@code (2, 4, 3)}. Previously modeled as a first-argument {@code
   * pass_through}, which reported the input shape unchanged. See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTransposePerm()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_transpose.py",
        "consume_transpose_perm",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_4_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.transpose} resolves a {@code perm} passed as a tensor constant (rather
   * than a Python list literal): a {@code (2, 3, 4)} input with {@code perm = tf.constant([2, 1,
   * 0])} permutes precisely to {@code (4, 3, 2)}. Exercises the tensor-constant {@code perm}
   * resolution path of {@link com.ibm.wala.cast.python.ml.client.Transpose}, distinct from the
   * list-literal path. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket
   * 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTransposeTensorPerm()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_transpose.py",
        "consume_transpose_tensor_perm",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_3_2_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.diag(diagonal)} increases rank by one: it places the input's
   * last axis on the diagonal of a new trailing square, so a {@code (4,)} input yields {@code (4,
   * 4)}. Previously modeled as a first-argument {@code pass_through}, which reported the input
   * shape unchanged. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDiag()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_linalg_rank_delta.py",
        "consume_diag",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.diag_part(input)} decreases rank by one: it extracts the
   * diagonal of each trailing square, so a {@code (3, 3)} input yields {@code (3,)}. Previously
   * modeled as a first-argument {@code pass_through}, which reported the input shape unchanged. See
   * <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDiagPart()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_linalg_rank_delta.py",
        "consume_diag_part",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.matrix_transpose(a)} swaps the last two axes, preserving leading
   * batch dimensions, so a {@code (2, 3)} input yields {@code (3, 2)}. Previously modeled as a
   * first-argument {@code pass_through}, which reported the input shape unchanged. See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testMatrixTranspose()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_linalg_rank_delta.py",
        "consume_matrix_transpose",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.adjoint(matrix)} (conjugate transpose) swaps the last two axes
   * exactly like {@code matrix_transpose}, so a {@code (2, 3)} input yields {@code (3, 2)}.
   * Previously modeled as a first-argument {@code pass_through}, which reported the input shape
   * unchanged. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testAdjoint()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_linalg_rank_delta.py",
        "consume_adjoint",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.tile(input, multiples)} multiplies each axis by the corresponding entry
   * of {@code multiples}, so a {@code (2, 3)} input tiled by {@code [2, 1]} yields {@code (4, 3)}.
   * Previously modeled as a first-argument {@code pass_through}, which reported the input shape
   * unchanged. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTile()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tile.py", "consume_tile", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.solve(matrix, rhs)} reports the shape and dtype of the
   * right-hand side {@code rhs} (arg 1), not the coefficient {@code matrix} (arg 0). A {@code (3,
   * 3)} matrix and a {@code (3, 5)} rhs yield a {@code (3, 5)} result. Previously the op was
   * modeled as a first-argument {@code pass_through}, which reported the {@code (3, 3)} matrix
   * shape — an actively wrong answer. See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2b.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSolve()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_solve.py", "consume_solve", 1, 1, Map.of(2, Set.of(TENSOR_3_5_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.cholesky_solve(chol, rhs)} reports the shape and dtype of the
   * right-hand side {@code rhs} (arg 1), not the Cholesky factor {@code chol} (arg 0). See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2b.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCholeskySolve()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_solve.py", "consume_cholesky_solve", 1, 1, Map.of(2, Set.of(TENSOR_3_5_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.triangular_solve(matrix, rhs)} reports the shape and dtype of
   * the right-hand side {@code rhs} (arg 1), not the coefficient {@code matrix} (arg 0). See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2b.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTriangularSolve()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_solve.py",
        "consume_triangular_solve",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_5_FLOAT32)));
  }

  /**
   * Tier-6 op (wala/ML#449): {@code tf.sort(values, ...)}. The XML routes the call through {@code
   * convert_to_tensor} of {@code values}, so shape and dtype pass through unchanged — no dedicated
   * generator needed.
   */
  @Test
  public void testSort()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sort.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_6_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.tensor_scatter_nd_update}. Output dtype AND shape are
   * inherited from the {@code tensor} input — true shape-and-dtype passthrough on the first arg.
   * Here the input is shape {@code (4,)} float32, so the precise expected result is {@code (4,)
   * float32}. See {@link com.ibm.wala.cast.python.ml.client.TensorScatterNdUpdate} (wala/ML#449).
   *
   * <p>Post-master-merge of wala/ML#380's `tensor_scatter_nd_update` inlining (#237), the analysis
   * also produces an additional fully-⊤ context — likely a context where the input arg's PTS isn't
   * recovered through the inlined synthetic body, so the passthrough returns ⊤/UNKNOWN. The
   * assertion captures both contexts (per the prefer-observed-assertion convention from
   * `CONTRIBUTING.md`); a precision improvement that eliminates the ⊤ context will narrow the
   * actual to just {@code TENSOR_4_FLOAT32} and this test will fail with a clear "expected union,
   * got per-context" diff — that's the cue to update.
   *
   * <p>TODO(wala/ML#474): Once the additional fully-⊤ context for the inlined-passthrough path is
   * investigated/eliminated, narrow the assertion to {@code Set.of(TENSOR_4_FLOAT32)}.
   */
  @Test
  public void testTensorScatterNdUpdate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tensor_scatter_nd_update.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_FLOAT32, TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Generator-dispatch test for {@code tf.sequence_mask}. With no {@code dtype} argument the output
   * dtype is the TF-default {@code bool}; with a constant {@code maxlen} the shape is {@code
   * lengths.shape + [maxlen]}, so {@code sequence_mask([1, 3, 2], maxlen=5)} yields (3, 5).
   * (wala/ML#449 Tier 8.)
   */
  @Test
  public void testSequenceMask()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sequence_mask.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_5_BOOL)));
  }

  /**
   * Generator-dispatch test for {@code tf.sequence_mask} with an explicit {@code dtype} override
   * ({@code tf.sequence_mask(..., maxlen=5, dtype=tf.int32)}). The output dtype follows the
   * argument ({@code int32}) rather than the default {@code bool}, and the constant {@code maxlen}
   * gives the precise (3, 5) shape. Regression guard for surfacing the {@code dtype} parameter
   * through {@link com.ibm.wala.cast.python.ml.client.SequenceMask}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSequenceMaskDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sequence_mask_dtype.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_5_INT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.nn.embedding_lookup}. Output dtype is inherited from the
   * {@code params} input (here float32); the output shape is {@code ids.shape + params.shape[1:]},
   * so a (3, 2) table looked up by (2,) ids yields (2, 2). See {@link
   * com.ibm.wala.cast.python.ml.client.EmbeddingLookup} (wala/ML#449 Tier 8).
   */
  @Test
  public void testEmbeddingLookup()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_embedding_lookup.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.gather_nd}. Output dtype is inherited from the {@code
   * params} input (here float32); the output shape is {@code indices.shape[:-1] +
   * params.shape[indices.shape[-1]:]}, so a (2, 2) table indexed by (2, 2) depth-2 indices yields
   * (2,). See {@link com.ibm.wala.cast.python.ml.client.GatherNd} (wala/ML#449 Tier 8).
   */
  @Test
  public void testGatherNd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gather_nd.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.boolean_mask}. Output dtype is inherited from the {@code
   * tensor} input (here float32); the masked axis collapses to a dynamic dimension (the runtime
   * {@code True} count), so masking a (3, 2) tensor with a rank-1 mask yields {@code [Dynamic, 2]}.
   * The leading dimension is inherently runtime, so it stays dynamic. See {@link
   * com.ibm.wala.cast.python.ml.client.BooleanMask} (wala/ML#449 Tier 8).
   */
  @Test
  public void testBooleanMask()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_boolean_mask.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.image.extract_patches}. Output dtype is inherited from
   * the {@code images} input (here float32); the output shape is {@code [batch, out_rows, out_cols,
   * sizes_r * sizes_c * channels]}, so a (1, 10, 10, 3) image with {@code sizes=[1, 3, 3, 1]},
   * {@code strides=[1, 5, 5, 1]}, {@code rates=[1, 1, 1, 1]} and {@code VALID} padding yields (1,
   * 2, 2, 27). See {@link com.ibm.wala.cast.python.ml.client.ExtractPatches} (wala/ML#449 Tier 8).
   */
  @Test
  public void testExtractPatches()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_extract_patches.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_2_2_27_FLOAT32)));
  }

  /**
   * Regression guard for {@code tf.image.extract_patches} called with a Python <em>list
   * literal</em> {@code images} argument (rather than a {@code tf.Tensor}), per <a
   * href="https://github.com/wala/ML/issues/584">wala/ML#584</a>. The list-literal shape {@code (1,
   * 1, 1, 1)} is recovered from the nesting structure, and the result is the concrete {@code (1, 0,
   * 0, 9) int32}: a {@code 3x3} patch does not fit the {@code 1x1} image, so {@code VALID} padding
   * yields a 0-extent spatial output (depth {@code 3*3*1 = 9}), matching the runtime shape the
   * Python fixture asserts (<a href="https://github.com/wala/ML/issues/585">wala/ML#585</a>).
   */
  @Test
  public void testExtractPatches2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_extract_patches2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_0_0_9_INT32)));
  }

  /**
   * Regression guard for {@code tf.image.extract_patches} called with an {@code images} argument
   * built from a nested list <em>comprehension</em> (the wala/ML#584 corpus case), per <a
   * href="https://github.com/wala/ML/issues/584">wala/ML#584</a>. Resolving such an {@code images}
   * operand throws inside the generator; before the fix that aborted the whole type computation and
   * the result dropped its tensor classification entirely. The result must still be recognized as a
   * tensor — here ⊤ shape and ⊤ dtype, since a comprehension's computed elements (unlike a
   * literal's constants) yield no statically inferable dtype.
   */
  @Test
  public void testExtractPatchesComprehension()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_extract_patches_comprehension.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /** Pure-passthrough generator test for {@code tf.math.tan} (wala/ML#422). */
  @Test
  public void testTan()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tan.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.asin} (wala/ML#422). */
  @Test
  public void testAsin()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_asin.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.atan} (wala/ML#422). */
  @Test
  public void testAtan()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_atan.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.sinh} (wala/ML#422). */
  @Test
  public void testSinh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sinh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.cosh} (wala/ML#422). */
  @Test
  public void testCosh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_cosh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.asinh} (wala/ML#422). */
  @Test
  public void testAsinh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_asinh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.acosh} (wala/ML#422). */
  @Test
  public void testAcosh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_acosh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.atanh} (wala/ML#422). */
  @Test
  public void testAtanh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_atanh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.log1p} (wala/ML#422). */
  @Test
  public void testLog1p()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_log1p.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.expm1} (wala/ML#422). */
  @Test
  public void testExpm1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_expm1.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.round} (wala/ML#422). */
  @Test
  public void testRound()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_round.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.reciprocal} (wala/ML#422). */
  @Test
  public void testReciprocal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reciprocal.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.softplus} (wala/ML#422). */
  @Test
  public void testSoftplus()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_softplus.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.softsign} (wala/ML#422). */
  @Test
  public void testSoftsign()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_softsign.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.square} (wala/ML#422). */
  @Test
  public void testSquare()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_square.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.erf} (wala/ML#422). */
  @Test
  public void testErf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_erf.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.erfc} (wala/ML#422). */
  @Test
  public void testErfc()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_erfc.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Elementwise binary generator test for {@code tf.math.atan2} (wala/ML#422). */
  @Test
  public void testAtan2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_atan2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Elementwise binary generator test for {@code tf.math.maximum} (wala/ML#422). */
  @Test
  public void testMaximum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_maximum.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Elementwise binary generator test for {@code tf.math.minimum} (wala/ML#422). */
  @Test
  public void testMinimum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_minimum.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testAtan2}: {@code tf.math.atan2(y=..., x=...)}. Exercises
   * the kw-arg-resolution path on the {@code ElementWiseOperation} dispatch.
   */
  @Test
  public void testAtan2Kw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_atan2_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testMaximum}: {@code tf.math.maximum(x=..., y=...)}.
   * Exercises the kw-arg-resolution path on the {@code ElementWiseOperation} dispatch.
   */
  @Test
  public void testMaximumKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_maximum_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testMinimum}: {@code tf.math.minimum(x=..., y=...)}.
   * Exercises the kw-arg-resolution path on the {@code ElementWiseOperation} dispatch.
   */
  @Test
  public void testMinimumKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_minimum_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.fashion_mnist.load_data()}. Shapes and dtype are
   * identical to {@code mnist.load_data()}. The fixture passes all four unpacked arrays ({@code
   * x_train}, {@code y_train}, {@code x_test}, {@code y_test}) into the 4-arg sink, so the
   * assertion pins types at {@code vn=2..5}.
   */
  @Test
  public void testFashionMnistLoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_fashion_mnist_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_60000_28_28_UINT8),
            3, Set.of(TENSOR_60000_UINT8),
            4, Set.of(TENSOR_10000_28_28_UINT8),
            5, Set.of(TENSOR_10000_UINT8)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.cifar100.load_data()}. Shapes are identical to
   * {@code cifar10.load_data()}, but the {@code y_train} / {@code y_test} dtype is {@code int64}
   * (cifar100's class indices) rather than {@code uint8} (cifar10's class indices). The dispatch
   * routes through the dedicated {@link com.ibm.wala.cast.python.ml.client.Cifar100InputData}
   * generator (closes wala/ML#487's mis-routing through {@code Cifar10InputData}). Asserts on all
   * four unpacked arrays at {@code vn=2..5}.
   */
  @Test
  public void testCifar100LoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_cifar100_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_50000_32_32_3_UINT8),
            3, Set.of(TENSOR_50000_1_INT64),
            4, Set.of(TENSOR_10000_32_32_3_UINT8),
            5, Set.of(TENSOR_10000_1_INT64)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.reuters.load_data()}. Asserts on all four unpacked
   * arrays at {@code vn=2..5}: {@code x_train} ({@code (8982,)} {@code object} &mdash; newswires
   * are variable-length integer-encoded sequences, so numpy stores them in an {@code object}
   * array), {@code y_train} ({@code (8982,)} {@code int64}), {@code x_test} ({@code (2246,)} {@code
   * object}), {@code y_test} ({@code (2246,)} {@code int64}). The {@code object} dtype matches the
   * runtime truth the Python fixture asserts (wala/ML#488).
   */
  @Test
  public void testReutersLoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reuters_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_8982_OBJECT),
            3, Set.of(TENSOR_8982_INT64),
            4, Set.of(TENSOR_2246_OBJECT),
            5, Set.of(TENSOR_2246_INT64)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.boston_housing.load_data()}. Asserts on all four
   * unpacked arrays at {@code vn=2..5}: {@code x_train} ({@code (404, 13)} {@code float64}
   * features), {@code y_train} ({@code (404,)} {@code float64} regression targets), {@code x_test}
   * ({@code (102, 13)} {@code float64}), {@code y_test} ({@code (102,)} {@code float64}).
   */
  @Test
  public void testBostonHousingLoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_boston_housing_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_404_13_FLOAT64),
            3, Set.of(TENSOR_404_FLOAT64),
            4, Set.of(TENSOR_102_13_FLOAT64),
            5, Set.of(TENSOR_102_FLOAT64)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.imdb.load_data()}. The four returned arrays each
   * have shape {@code (25000,)}: the {@code x_train} / {@code x_test} arrays carry numpy {@code
   * object} dtype (variable-length integer-encoded sequences, so numpy stores them in an {@code
   * object} array); the {@code y_train} / {@code y_test} arrays have dtype {@code int64} (binary
   * labels). Asserts on all four unpacked arrays at {@code vn=2..5}. The {@code object} dtype
   * matches the runtime truth the Python fixture asserts (wala/ML#488).
   */
  @Test
  public void testImdbLoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_imdb_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_25000_OBJECT),
            3, Set.of(TENSOR_25000_INT64),
            4, Set.of(TENSOR_25000_OBJECT),
            5, Set.of(TENSOR_25000_INT64)));
  }

  /**
   * Generator test for {@code tf.strings.as_string} on a 2-arg sink {@code f(y, x)}. Output shape
   * is the input's shape (here {@code (3,)}); output dtype is fixed to {@code string}
   * (wala/ML#422). Asserts that both the {@code as_string} output (`y`, string dtype) at vn=2 and
   * its input (`x`, float32) at vn=3 classify precisely &mdash; the multi-tensor-sink pattern
   * doesn't break classification on either flow when run inside the full test suite.
   */
  @Test
  public void testAsString2ArgSink()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_as_string_2arg_sink.py",
        "f",
        2,
        2,
        Map.of(2, Set.of(TENSOR_3_STRING), 3, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Sibling of {@link #testAsString2ArgSink()} with two 1-arg sinks {@code f(y)} and {@code g(x)},
   * asserting the {@code y}-side sink. The {@code as_string} output's string-dtype tensor
   * classifies precisely at vn=2.
   */
  @Test
  public void testAsStringTwoSinksY()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_as_string_two_sinks.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_STRING)));
  }

  /**
   * Companion to {@link #testAsStringTwoSinksY()} asserting the {@code x}-side sink {@code g(x)}.
   * The {@code x} input classifies precisely as float32 at vn=2.
   */
  @Test
  public void testAsStringTwoSinksX()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_as_string_two_sinks.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Regression test for `wala/ML#447`: scalar `tf.constant(<bool>)` exercises the {@link
   * java.lang.Boolean} arm of {@code TensorGenerator.getDTypesOfValue}. Without it, dtype inference
   * threw {@code IllegalStateException: Unknown constant type: class java.lang.Boolean}.
   */
  @Test
  public void testBoolConstant()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_bool_constant.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_BOOL)));
  }

  /**
   * List-of-bool form of {@link #testBoolConstant} — exercises the recursive {@code
   * getDTypesOfValue} call (line 1625) on a list whose elements are `Boolean` constants.
   */
  @Test
  public void testBoolConstant2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_bool_constant.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_3_BOOL)));
  }

  @Test
  public void testGradient()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gradient.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_FLOAT32)));
  }

  @Test
  public void testGradient2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gradient2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_FLOAT32)));
  }

  /**
   * Regression test for <a href="https://github.com/wala/ML/issues/464">wala/ML#464</a>: when
   * {@code sources} is a list (the common Keras pattern), {@code tape.gradient} returns a parallel
   * list of fresh tensors and {@code grads[i]} must resolve to the shape/dtype of the i-th source.
   * The fixture passes both {@code grads[0]} (for {@code w1}, a {@code [2]}-shaped float32) and
   * {@code grads[1]} (for {@code w2}, a {@code [1, 1]}-shaped float32) to {@code f} across two
   * separate calls; with the {@link com.ibm.wala.cast.python.ml.client.Gradient} {@code
   * TupleElementProvider} implementation, {@code f}'s parameter resolves to the union of {@link
   * #TENSOR_2_FLOAT32} and {@link #TENSOR_1_1_FLOAT32}.
   */
  @Test
  public void testGradientList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gradient_list.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_FLOAT32, TENSOR_1_1_FLOAT32)));
  }

  /**
   * Tighter variant of {@link #testGradientList()}: passes both gradients in a single {@code
   * f(grads[0], grads[1])} call, so the analyzer must resolve each argument's tensor type
   * independently per its source index rather than as a union across two call sites. {@code f}'s
   * first parameter (vn=2) must resolve to {@link #TENSOR_2_FLOAT32} (from {@code w1}) and the
   * second parameter (vn=3) to {@link #TENSOR_1_1_FLOAT32} (from {@code w2}). Closes part of <a
   * href="https://github.com/wala/ML/issues/464">wala/ML#464</a>.
   */
  @Test
  public void testGradientList2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gradient_list2.py",
        "f",
        2,
        2,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_1_1_FLOAT32)));
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
   * Operands of different ranks ({@code (2, 3)} and {@code (2,)}) are genuinely non-broadcastable.
   * Rather than throw an exception that aborts the whole analysis, the element-wise generator
   * degrades the result shape to ⊤ (unknown) and continues; the {@code int32} dtype is still
   * recovered (<a href="https://github.com/wala/ML/issues/583">wala/ML#583</a>).
   *
   * <p>Here ⊤ is the correct final result — incompatible operands have no valid broadcast shape —
   * so, unlike the recoverable list-literal case ({@link #testExtractPatches2}), there is no
   * precision to recover and no shape-tightening TODO.
   */
  @Test
  public void testMultiply6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_multiply6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  @Test
  public void testMultiply7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_multiply7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testRelu()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_relu.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  // Tier-A pure-passthrough math ops (wala/ML#422). Each is shape and dtype passthrough on `x`;
  // routes to a per-op subclass of `PassThroughUnaryTensorGenerator`. The point of dedicated
  // generators (vs. leaving these on `ReadDataFallback`) is dtype propagation: without these,
  // `tf.math.sqrt(x)` etc. produce ⊤/UNKNOWN, blocking downstream dtype-axis precision through
  // any function whose parameters flow from these ops.

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Sqrt}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSqrt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sqrt.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Log}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLog()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_log.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Negative}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testNegative()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_negative.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Sin}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSin()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sin.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Cos}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCos()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_cos.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Floor}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testFloor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_floor.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Sign}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSign()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sign.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.cast(x, dtype)}. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Cast} generator extends {@link
   * com.ibm.wala.cast.python.ml.client.PassThroughUnaryTensorGenerator} for shape and overrides the
   * dtype-arg position to point at {@code dtype}; the {@code tf.cast} {@code pass_through} alias
   * that previously bypassed the override was removed in <a
   * href="https://github.com/wala/ML/issues/499">wala/ML#499</a>, so the static analysis now
   * reports the explicit cast target ({@code int32}) rather than the input's dtype ({@code
   * float32}).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCast()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_cast.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/509">wala/ML#509</a>: a
   * user-defined class that happens to define a {@code set_shape} method must not be classified as
   * a tensor by the static analysis. The {@code set_shape} recognition path must restrict pinning
   * to actual tensor types and let non-tensor receivers fall through untouched.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSetShapeNonTensorReceiver()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_set_shape_non_tensor.py", "consume", 0, 0, Map.of());
  }

  /**
   * Generator-dispatch test for {@code tf.expand_dims(input, axis)}. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.ExpandDims} generator overrides {@code getDefaultShapes} to
   * ⊤ pending an axis-aware shape composer. Replacing the stale {@code array_ops.expand_dims}
   * pass_through alias with the dedicated routing (this PR's earlier review fix) made the override
   * actually fire, so the assertion now sees the ⊤ output the override emits.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/481">wala/ML#481</a>): once the axis-aware
   * composer lands as a follow-up, tighten this from {@code TENSOR_UNKNOWN_SHAPE_FLOAT32} to {@code
   * (1, 3)} float32 (the precise insert-at-axis result).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testExpandDims()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_expand_dims.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testExpandDims()} for {@code axis=-1}: trailing length-1 dim. Input shape
   * {@code (3,)} produces output shape {@code (3, 1)}.
   */
  @Test
  public void testExpandDimsAxisNeg1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_expand_dims_axis_neg1.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_1_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.clip_by_value(t, clip_value_min, clip_value_max)}. Pure
   * passthrough — output shape and dtype both inherit from {@code t}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testClipByValue()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_clip_by_value.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.nn.leaky_relu(features)}. Pure passthrough — output shape
   * and dtype both inherit from {@code features} (the input tensor). Mirrors {@link #testRelu()};
   * the {@link com.ibm.wala.cast.python.ml.client.LeakyRelu} generator extends {@link
   * com.ibm.wala.cast.python.ml.client.PassThroughUnaryTensorGenerator}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLeakyRelu()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_leaky_relu.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.math.pow(x, y)}. Element-wise binary; output shape is the
   * broadcast of {@code x} and {@code y} (here both {@code (3,)}, so {@code (3,)}); output dtype
   * matches {@code x} (TF requires {@code x}/{@code y} to share dtype, so dtype-from-{@code x} is
   * sound). Routed through {@link com.ibm.wala.cast.python.ml.client.ElementWiseOperation}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPow()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_pow.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testPow}: {@code tf.math.pow(x=x, y=y)}. Exercises the
   * keyword arg-resolution path through {@link
   * com.ibm.wala.cast.python.ml.client.ElementWiseOperation}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPowKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_pow_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Mixed positional/keyword variant of {@link #testPow}: {@code tf.math.pow(x, y=y)}. Exercises
   * the case where the first argument is positional and the rest are keyword arguments.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPowMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_pow_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
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
   * Regression guard for the {@code @dataclass} half of <a
   * href="https://github.com/wala/ML/issues/205">wala/ML#205</a>: a module containing a {@code
   * @dataclass} definition loads and analyzes without a front-end parse error. The dataclass is
   * defined but unused in the dataflow; {@code f} receives a tensor directly, so its parameter type
   * is recovered iff the module parsed. Companion to {@link #testModule68}/{@link #testModule69},
   * which guard the same for {@code NamedTuple}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDataclassParse()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataclass_parse.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Verifies tensor types propagate through a user-defined {@code NamedTuple} field (<a
   * href="https://github.com/wala/ML/issues/579">wala/ML#579</a>): a tensor stored in a {@code
   * NamedTuple} field and read back ({@code b = w.tensor}) keeps its {@code (4, 8) float32} type.
   * Unlike {@link #testModule68}/{@link #testModule69} — which only confirm a {@code NamedTuple}
   * <em>definition</em> parses — this exercises actual field dataflow: PEP-526 annotated fields now
   * reach the CAst as ordered field entities (jython3 grammar/AST support), and the synthesized
   * constructor populates them positionally. It is the minimal form of the GCN blocker in
   * wala/ML#570, where {@code GraphConvolution.call} unwraps a {@code GNNInput} {@code NamedTuple}
   * the same way.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNamedTupleFieldRead()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_namedtuple_field.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins the forward result of a nested layer call ({@code self.inner(...)} inside another layer's
   * {@code call}): the inner layer's return is tensor-typed at the nested call site and flows into
   * a sink (<a href="https://github.com/wala/ML/issues/570">wala/ML#570</a>; enabled by the
   * wala/ML#595 forward-result machinery).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNestedLayerCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nested_layer_call.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the forward result of a nested layer call whose argument is a {@code NamedTuple} and whose
   * inner {@code call} computes through a field read, a matmul, and an {@code unsorted_segment_sum}
   * (<a href="https://github.com/wala/ML/issues/570">wala/ML#570</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNestedLayerCallNamedTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nested_layer_call2.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins the forward result of a nested layer call whose inner {@code call} returns through a
   * second method hop ({@code self.propagate(...)} on the same class), the single-class form of
   * {@code gcn_proj}'s return chain (<a href="https://github.com/wala/ML/issues/570">
   * wala/ML#570</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNestedLayerCallPropagate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nested_layer_call3.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins the forward result of a nested layer call whose inner {@code call} returns through an
   * <em>inherited</em> method ({@code self.propagate(...)} defined on a same-module base class),
   * mirroring {@code gcn_proj}'s {@code GraphConvolution(MessagePassing)} shape (<a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNestedLayerCallInheritedPropagate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nested_layer_call4.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins the forward result of a nested layer call whose inner {@code call} returns through a
   * method inherited from a <em>cross-module</em> base ({@code Inner(MessagePassing)} with {@code
   * MessagePassing} in another file), the deepest structural form of the {@code gcn_proj} chain (<a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a>). {@code consume(x)} inside {@code
   * Outer.call} recovers the concrete {@code (4, 8) float32}, so the frame/inheritance mechanism is
   * not the {@code gcn_proj} residual; the remaining gap is the list-mediated aggregation inside
   * the vendored {@code propagate} (gathers/slices/appends over the adjacency-list collection).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNestedLayerCallCrossModule()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "nested_proj/messagepassing.py",
          "nested_proj/inner.py",
          "nested_proj/tf2_test_nested_cross_module.py"
        },
        "tf2_test_nested_cross_module.py",
        "consume",
        "nested_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins the {@code add_weight} result itself (wala/ML#667): the weight's shape comes from the
   * {@code shape} list argument and its dtype from a {@code tf.float32} module-constant argument
   * (the string form is covered by {@link #testCollectionProbeAddWeight()}).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testAddWeightArguments()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add_weight2.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/672">wala/ML#672</a>: {@code add_weight}
   * without a {@code dtype} argument follows Keras's documented default and types float32 (the
   * layer variable dtype under the default global policy), via the allocator's float32 default.
   * Completes the dtype-form trio with {@link #testCollectionProbeAddWeight()} (string) and {@link
   * #testAddWeightArguments()} (module constant).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testAddWeightDefaultDtype()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add_weight3.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/683">wala/ML#683</a>: {@code
   * self._distribution_strategy} is Keras-internal state assigned by {@code Model.__init__}, never
   * in user code. With the shell {@code Model.__init__} modeling the attribute, the receiver of
   * {@code self._distribution_strategy.run(self.__train_step, args=(x, y))} resolves to the
   * strategy instance and the {@code run} summary materializes the invoke edge, typing both
   * callback parameters through the args-tuple forwarding (wala/ML#618).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDistTrainStep()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dist_train_step.py",
        "MyModel.__train_step",
        2,
        3,
        Map.of(3, Set.of(TensorType.of(FLOAT_32, 2, 3)), 4, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/683">wala/ML#683</a> at subject shape
   * (MusicTransformer-tensorflow2.0): the model base is {@code keras.Model} bound by {@code from
   * tensorflow.python import keras}, so the summary {@code Model} must be reachable through the
   * {@code tensorflow.python} module object for the class shell to carry {@code Model.__init__}'s
   * {@code _distribution_strategy} assignment; and the callback args tuple has four elements, so
   * the strategy {@code run} summary must forward past the first two.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDistTrainStep2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dist_train_step2.py",
        "MyModel.__train_step",
        3,
        4,
        Map.of(
            3,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            4,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            5,
            Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/683">wala/ML#683</a> at the
   * MusicTransformer-tensorflow2.0 encoder-decoder shape: the callback args tuple has seven
   * elements, the widest the subject passes to the strategy, so the {@code run} summary's tuple
   * forwarding must reach fields 2 through 6. The fixture's callback computes from the tuple's
   * fifth and sixth elements specifically, so the pinned result only types if the later fields
   * flow.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDistTrainStep3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dist_train_step3.py",
        "MyModel.__train_step",
        6,
        7,
        Map.of(
            3,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            4,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            5,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            6,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            7,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            8,
            Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/684">wala/ML#684</a> at subject scale: in the
   * vendored MusicTransformer-tensorflow2.0 whole-project analysis, {@code
   * MusicTransformer.__prepare_train_data}'s direct {@code tf.ones((y.shape[0], 1), dtype=y.dtype)}
   * must be visible through the {@code tf} binding carried by {@code from custom.layers import *}
   * (a wildcard re-export of an import binding). What is asserted is the function's tensor-variable
   * census: seven distinct value numbers, which include the {@code tf.ones} result and its
   * downstream locals (observed as the runtime-true {@code (Dynamic, 1)} of float32 at vn 6);
   * pre-fix, only the two ⊤-typed parameters survived, so any regression of the binding collapses
   * the count. The vendored file is verbatim, so no {@code consume} sink can pin the local's exact
   * type here; the exact-type pins live in the fixture-scale probes ({@link #testWildcardUsedTf()},
   * {@link #testWildcardPkgNoInitTf()}). (The sibling {@code
   * MusicTransformerDecoder.__prepare_train_data} has no live {@code tf} use; all its tensor lines
   * are commented out in the subject.) Before the wala/ML#683 {@code tensorflow.python} namespace
   * binding, the empty {@code keras} binding that {@code custom/layers.py} re-exports starved the
   * whole chain.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   *     <p>The local count excludes the two all-constant slice-constructor objects under the {@code
   *     [:, :-1]} subscripts, pinned non-tensor since wala/ML#732.
   */
  @Test
  public void testMusicTransformerPrepareTrainData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        MUSICTRANSFORMER_PROJECT_FILES,
        "model.py",
        "MusicTransformer.__prepare_train_data",
        "musictransformer_proj",
        2,
        5,
        Map.of(
            2,
            Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE),
            3,
            Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Pins wala/ML#670's fixes directly: {@code GlobalAveragePooling1D} is modeled (rank-3 input,
   * temporal axis dropped), so the functional model's weight walk resolves the downstream {@code
   * Dense} kernel {@code (8, 5)} and bias {@code (5,)} concretely.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGap1dWeights()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gap1d_weights.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 8, 5), TENSOR_5_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/669">wala/ML#669</a>: {@code
   * build_model} is vendored verbatim from {@code LongmaoTeamTf/deep_recommenders} ({@code
   * examples/train_transformer_on_imdb_keras.py}), a functional {@code tf.keras.Model} whose
   * weight-graph walk ({@code Model.getWeightShapes}) resolves {@code Dense}/{@code MatMul} weight
   * shapes — the exact frames that crashed with {@code IllegalStateException} on 0.52.12 when a
   * WALA 1.8.0 non-constant key reached {@code getConstantValues}. The crash's non-constant {@code
   * units} key itself arises only under the consumer-side speculative call-graph configuration,
   * which this harness does not enable, so this guard pins that the walk completes; the {@code
   * getConstantValues} degrade contract is exercised structurally.
   *
   * <p>With wala/ML#670 fixed, the walk traverses past the head {@code Dense} into the transformer
   * (an unresolvable input no longer stops the trace-back, and {@code GlobalAveragePooling1D} is
   * modeled — see {@link #testGap1dWeights()}), but it still yields no weight shapes here: the
   * pooling input is the vendored transformer's forward output, whose shape is the wala/ML#570
   * residual. TODO: expect the concrete weight-shape union once <a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a> is fixed.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTransformerWeights()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "tr_proj/deep_recommenders/__init__.py",
          "tr_proj/deep_recommenders/keras/__init__.py",
          "tr_proj/deep_recommenders/keras/models/__init__.py",
          "tr_proj/deep_recommenders/keras/models/nlp/__init__.py",
          "tr_proj/deep_recommenders/keras/models/nlp/multi_head_attention.py",
          "tr_proj/deep_recommenders/keras/models/nlp/transformer.py",
          "tr_proj/tf2_test_transformer_weights.py"
        },
        "tf2_test_transformer_weights.py",
        "consume",
        "tr_proj",
        0,
        0,
        emptyMap());
  }

  /**
   * Control half of the <a href="https://github.com/wala/ML/issues/687">wala/ML#687</a> MRE: the
   * sibling script's Keras layer reached through {@code from B import Padding2D} analyzes fully —
   * the layer call's result types concretely.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testImportFrom()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"importmod_proj/tf2_test_import_from.py", "importmod_proj/B.py"},
        "B.py",
        "Padding2D.call",
        "importmod_proj",
        1,
        1,
        Map.of(
            3,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        DynamicDim.INSTANCE,
                        new NumericDim(32),
                        new NumericDim(32),
                        new NumericDim(3))))));
  }

  /**
   * Reported-failing half of the <a href="https://github.com/wala/ML/issues/687">wala/ML#687</a>
   * MRE: the byte-identical layer reached through a plain {@code import B} module object, with the
   * importer passed <em>first</em> — the translation order that reproduced the loss before the
   * scope-membership binding fix (<a href="https://github.com/wala/ML/issues/691">wala/ML#691</a>):
   * the plain-import binding used to require the importee to be already translated.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testImportModule()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"importmod_proj/tf2_test_import_module.py", "importmod_proj/B.py"},
        "B.py",
        "Padding2D.call",
        "importmod_proj",
        1,
        1,
        Map.of(
            3,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        DynamicDim.INSTANCE,
                        new NumericDim(32),
                        new NumericDim(32),
                        new NumericDim(3))))));
  }

  /**
   * Importee-first twin of {@link #testImportModule()} (wala/ML#691): the previously-working
   * translation order, guarded so both orders stay equivalent.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testImportModuleImporteeFirst()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"importmod_proj/B.py", "importmod_proj/tf2_test_import_module.py"},
        "B.py",
        "Padding2D.call",
        "importmod_proj",
        1,
        1,
        Map.of(
            3,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        DynamicDim.INSTANCE,
                        new NumericDim(32),
                        new NumericDim(32),
                        new NumericDim(3))))));
  }

  /**
   * Probe for <a href="https://github.com/wala/ML/issues/684">wala/ML#684</a>: {@code tf} arrives
   * through {@code from helpers import *} in a script with no direct tensorflow import, and is read
   * inside a name-mangled {@code @staticmethod} of a {@code tf.keras.Model} subclass invoked
   * self-qualified — the subject's {@code MusicTransformer.__prepare_train_data} shape, several
   * levels deeper than {@link #testCollectionProbeWildcardTf()}'s script-level read. The wildcard
   * binding resolves, the shape is concrete, and the {@code dtype=y.dtype} attribute argument is
   * consumed (wala/ML#686), so the result is the runtime-true {@code (2, 1) int32}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testWildcardMethodTf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"wildcard_proj/helpers.py", "wildcard_proj/tf2_test_wildcard_method_tf.py"},
        "tf2_test_wildcard_method_tf.py",
        "consume",
        "wildcard_proj",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 2, 1))));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/684">wala/ML#684</a> at fixture scale: {@code
   * tf} reached through {@code from helpers_used import *} where the exporting module also reads
   * {@code tf} inside one of its own functions. The intra-module use lexically exposes the binding,
   * which drops it from the script body's SSA local names, so the wildcard scan's named-binding
   * match (wala/ML#665) must consult the exposed-name information as well. The subject's {@code
   * custom/layers.py} has this shape; the untouched-binding {@code helpers.py} probes do not.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testWildcardUsedTf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "wildcard_proj/helpers_used.py", "wildcard_proj/tf2_test_wildcard_used_tf.py"
        },
        "tf2_test_wildcard_used_tf.py",
        "consume",
        "wildcard_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/684">wala/ML#684</a> at fixture scale for the
   * package-qualified form: {@code tf} reached through {@code from pkgnoinit.helpers3 import *},
   * where the package has no {@code __init__.py} (a namespace package, like the subject's {@code
   * custom/}) and the exporting module also reads {@code tf} in one of its own functions — the
   * exact {@code from custom.layers import *} shape of MusicTransformer-tensorflow2.0.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testWildcardPkgNoInitTf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "wildcard_proj/pkgnoinit/helpers3.py", "wildcard_proj/tf2_test_wildcard_pkgnoinit_tf.py"
        },
        "tf2_test_wildcard_pkgnoinit_tf.py",
        "consume",
        "wildcard_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the vendored {@code LayerNormalization} forward result: {@code add_weight}-created {@code
   * gamma}/{@code beta} dispatch (wala/ML#595, wala/ML#618) and the normalization body types.
   * Receiver-keyed contexts (wala/ML#679) dropped the shapeless-and-dtypeless union member, and
   * with parameter defaults materializing in the pointer analysis (wala/ML#743), the {@code
   * epsilon=1e-6} co-operand classifies as a runtime scalar, so the {@code variance + epsilon}
   * broadcast preserves the operand shape and the result is the runtime-true concrete {@code (2, 3,
   * 8)} float32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testVendoredLayerNorm()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_vendored/layers/__init__.py", "gpt2_vendored/layers/embedding_layer.py",
          "gpt2_vendored/layers/feed_forward.py", "gpt2_vendored/layers/layer_norm.py",
          "gpt2_vendored/layers/attention_layer.py", "gpt2_vendored/utils/__init__.py",
          "gpt2_vendored/utils/tf_utils.py", "gpt2_vendored/scripts/__init__.py",
          "gpt2_vendored/scripts/utils.py", "gpt2_vendored/data_pipeline.py",
          "gpt2_vendored/A.py", "gpt2_vendored/probe_ln.py"
        },
        "probe_ln.py",
        "consume",
        "gpt2_vendored",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3, 8))));
  }

  /**
   * Probes wala/ML#666's dotted-alias case ({@code import tensorflow.keras.backend as K} read
   * inside a method).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testBackendAliasCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_backend_alias.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  @Test
  public void testNamedTupleFieldMatmul()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_namedtuple_field_matmul.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Same as {@link #testNamedTupleFieldRead} but the {@code NamedTuple} base is written as the
   * dotted attribute chain {@code typing.NamedTuple} rather than a bare {@code NamedTuple}. This
   * guards the dotted-base path of {@code PythonConstructorTargetSelector.isPositionalFieldClass}:
   * the front-end must record the full {@code typing.NamedTuple} supertype name (not just the root
   * {@code typing}) for the positional-field synthesis to fire, so the tensor stored in the field
   * and read back ({@code b = w.tensor}) keeps its {@code (4, 8) float32} type. Without the full
   * dotted-name capture the supertype collapses to {@code typing}, {@code isPositionalFieldClass}
   * returns {@code false}, and {@code consume} sees zero tensor parameters (<a
   * href="https://github.com/wala/ML/issues/571">wala/ML#571</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNamedTupleFieldReadDotted()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_namedtuple_field_dotted.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Verifies tensor types propagate through a {@code typing.Tuple}-annotated tuple-of-tensors
   * parameter: a 2-tuple of tensors passed to {@code f} and unpacked ({@code x, y = inputs}) keeps
   * each element's type, so {@code consume(x)} sees {@code (4, 8) float32}. This mirrors the
   * perf-eval corpus's {@code deep_recommenders} {@code CIN.call(self, inputs: Tuple[tf.Tensor,
   * tf.Tensor])} — an {@code @tf.function}-decorated function the Hybridize tool refactors — and is
   * the tuple-parameter analogue of {@link #testNamedTupleFieldRead}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTupleParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tuple_param.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Verifies a module-level PEP-526 annotated assignment with a value (`t: tf.Tensor = tf.ones([2,
   * 3])`) declares its target and propagates the value (wala/ML#579). Outside a class body
   * `visitAnnAssign` must declare a simple-name target like `visitAssign` does; otherwise the
   * target is left undeclared. The tensor flows to `consume` as `(2, 3) float32`.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testAnnAssignLocal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_annassign_local.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
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

  /**
   * Test two call sites of the same method with equal total arity but different positional/keyword
   * splits. The trampoline cache must not collide them; see <a
   * href="https://github.com/wala/ML/issues/740">wala/ML#740</a>. The parameter {@code x} receives
   * a tensor positionally at one call site and by keyword at the other, so its type must be the
   * union of both.
   */
  @Test
  public void testTrampolineSplit() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_trampoline_split.py",
        "C.f",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32, TENSOR_1_3_FLOAT32)));
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

  /**
   * Tier-1 generator (wala/ML#449): {@code tf.identity(input)} returns a fresh tensor with the same
   * shape and dtype as {@code input}. Pre-fix this routed via {@code identity}'s synthetic XML
   * which ultimately allocates a {@code Ltensorflow/python/framework/ops/convert_to_tensor} —
   * {@link com.ibm.wala.cast.python.ml.client.ConvertToTensor} handles dtype/shape, but the extra
   * indirection through {@code identity}'s wrapper class can be tightened.
   */
  @Test
  public void testIdentity()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_identity.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Companion to {@link #testIdentity()} exercising the keyword-argument call site {@code
   * tf.identity(input=...)}. The arg-resolution helpers in {@link
   * com.ibm.wala.cast.python.ml.client.Identity} (and the underlying {@code
   * getArgumentPointsToSet(builder, paramPos, paramName)}) resolve keyword args via {@code
   * paramName}; without a kwarg fixture that branch is dead-on-arrival in the test data.
   */
  @Test
  public void testIdentityKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_identity_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Tier-1 generator (wala/ML#449): {@code tf.stop_gradient(input)} returns a fresh tensor with the
   * same shape and dtype as {@code input}. Pre-fix this routed through {@code ReadDataFallback}
   * (the alloc has no value/dtype field bindings, just a {@code read_data} marker) and emitted
   * {@code [{? of unknown}]}; the {@link com.ibm.wala.cast.python.ml.client.StopGradient} generator
   * now reads {@code input}'s shape/dtype directly via the same {@code shapesOfArg} / {@code
   * dtypesOfArg} pattern as {@link com.ibm.wala.cast.python.ml.client.Sigmoid}.
   */
  @Test
  public void testStopGradient()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_stop_gradient.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Companion to {@link #testStopGradient()} exercising the keyword-argument call site. */
  @Test
  public void testStopGradientKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_stop_gradient_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Lock-in test for wala/ML#449 Tier-1 coverage of {@code tf.nn.bias_add(value, bias)}: returns a
   * fresh tensor with the same shape and dtype as {@code value} (bias is broadcast-added but
   * doesn't change the receiver's shape). Unlike {@link #testIdentity()} / {@link
   * #testStopGradient()} — which got dedicated {@link com.ibm.wala.cast.python.ml.client.Identity}
   * / {@link com.ibm.wala.cast.python.ml.client.StopGradient} generators paired with a direct
   * {@code <new>+<return>} in the XML — {@code bias_add} is modeled in {@code tensorflow.xml} as a
   * delegation: {@code <new>} of a {@code convert_to_tensor} function-object, {@code <call>} of its
   * {@code do} with {@code value} as the argument, then {@code <return>} of that result (no
   * dedicated Java generator; the actual tensor allocation happens inside {@code
   * convert_to_tensor.do()}). That delegation suffices for the shape/dtype-passthrough semantics
   * this test exercises.
   */
  @Test
  public void testBiasAdd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_bias_add.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testBiasAdd()} exercising the all-keyword call site {@code
   * tf.nn.bias_add(value=..., bias=...)}.
   */
  @Test
  public void testBiasAddKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_bias_add_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Diagnostic for the existing {@code add_n} XML pattern (read list field 0 → {@code
   * convert_to_tensor}). Captures the observable static-analysis output for {@code tf.add_n([t1,
   * t2])} where both list elements are {@code tf.constant} tensors. If this test passes with {@code
   * TENSOR_3_INT32}, the list-element-PTS path through {@code <getfield class="Llist" field="0">}
   * works and Tier 5 ops ({@code concat}/{@code stack}/{@code meshgrid}) can use the same pattern
   * cheaply. If it produces {@code ? of unknown}, the pattern doesn't propagate types and Tier 5
   * needs Java-side list-element traversal logic.
   */
  @Test
  public void testAddN()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add_n.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Tier-5 generator (wala/ML#449): {@code tf.concat(values, axis)}. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Concat} generator computes the precise output shape by
   * walking every entry in the {@code values} list, summing each input's dim along the resolved
   * {@code axis}, and inheriting the rest of the shape from the first input. The fixture
   * concatenates two {@code (3,)} tensors along {@code axis=0}, so the precise output is {@code
   * (6,)}; dtype is inherited from the first element ({@code int32}).
   */
  @Test
  public void testConcat()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_concat.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_6_INT32)));
  }

  /**
   * Multi-rank {@code tf.concat([t1, t2], axis=1)} with {@code (2, 3)} inputs. Exercises the
   * rank-aware path in {@link com.ibm.wala.cast.python.ml.client.Concat#computeConcatenatedShape}:
   * non-axis dim preservation (the leading {@code 2} survives) and the axis-dim sum (the trailing
   * {@code 3 + 3 = 6}).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testConcatMultirank()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_concat_multirank.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_6_INT32)));
  }

  /**
   * {@code tf.concat([t1, t2], axis=-1)} with {@code (2, 3)} inputs. Exercises the negative-axis
   * normalization in {@link com.ibm.wala.cast.python.ml.client.Concat#computeConcatenatedShape}:
   * {@code axis = -1} resolves to {@code rank - 1 = 1} for rank-2 inputs, producing the same {@code
   * (2, 6)} answer as the explicit {@code axis=1} fixture.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testConcatNegativeAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_concat_negaxis.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_6_INT32)));
  }

  /**
   * Tier-5 generator (wala/ML#449): {@code tf.stack(values, axis)}. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Stack} generator computes the precise output shape by
   * reading the {@code values} list's PTS-derived length {@code N} and inserting it at the resolved
   * {@code axis} position into the first element's shape: {@code values[0].shape[:axis] + (N,) +
   * values[0].shape[axis:]}. The fixture stacks two {@code (3,)} tensors with {@code axis=0}, so
   * the precise output is {@code (2, 3)}; dtype is inherited from the first element ({@code
   * int32}).
   */
  @Test
  public void testStack()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_stack.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
  }

  /**
   * Distilled regression guard for the scalar-expression broadcast (wala/ML#718): a tensor scaled
   * by a statically opaque scalar expression ({@code 1.0 / math.sqrt(4.0)}) keeps its shape, since
   * the expression's scalarness is structural even though its value never resolves. The analysis
   * previously erased the result to ⊥. Mirrors NLPGNN's attention-logit scaling.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testScalarExpressionScale()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_scalar_scale.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * Distilled guard for the dimension-provenance split (wala/ML#721): a configuration-sourced size
   * — here an environment read the analysis cannot fold — types as {@link UnresolvedDim}, a fixed
   * runtime size of unknown value, not {@link DynamicDim}, which is reserved for axes the runtime
   * {@code TensorShape} reports as {@code None}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testUnresolvedDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_unresolved_dim.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(100))))));
  }

  /**
   * Pins the fold taint of the dimension-provenance split (wala/ML#721): {@code np.prod} over a
   * shape list whose leading axis is {@code None} stays {@link DynamicDim} — arithmetic over a
   * {@code None} axis is itself {@code None} at run time — rather than degrading to {@link
   * UnresolvedDim}. The runtime guards the {@code None} away before folding (the BERT {@code
   * get_shape_list} idiom), but the static walk sees the unguarded shape. The {@code (24)} member
   * is the guarded arm — the runtime-true fold over the patched {@code [1, 4, 6]} — flowing
   * alongside per the guard-φ.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testProdOverDynamic()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_prod_over_dynamic.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE)),
                TensorType.of(FLOAT_32, 24))));
  }

  /**
   * Distilled regression guard for {@code tf.matmul}'s batched form (wala/ML#718): the leading
   * (batch) dimensions carry through and the trailing two dimensions compose as the matrix product,
   * so the rank is preserved. The analysis previously collapsed every product to rank two.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBatchedMatMul()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_batched_matmul.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4, 5))));
  }

  /**
   * NLPGNN's {@code WDEmbedding} in miniature (wala/ML#711, wala/ML#717): the output reshape's
   * target slices the rank-conditional {@code tf.expand_dims} guard's φ, so the composition
   * cross-products the two source shapes per position: four members, including the runtime {@code
   * (2, 2, 8)}, with the trailing dimension static in every member. The mixed-pairing members are
   * the cross-product's sound over-approximation; pairing the evaluation per φ member would drop
   * them. The embedding table's {@code float32} survives through both the {@code gather} arm and
   * the {@code one_hot}/{@code matmul} arm.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEmbeddingOutput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_embedding_output.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 16),
                TensorType.of(FLOAT_32, 2, 8),
                TensorType.of(FLOAT_32, 2, 2, 16),
                TensorType.of(FLOAT_32, 2, 2, 8))));
  }

  /**
   * Opaque-size variant of {@link #testEmbeddingOutput()} (wala/ML#717), mirroring the vendored
   * NLPGNN embedding, whose table size comes from a checkpoint config the analysis cannot read. The
   * output reshape's trailing element {@code input_shape[-1] * self.embedding_size} then has an
   * unresolvable factor, but arithmetic over a shape-vector subscript is one scalar dimension, so
   * the value degrades to dynamic while the rank and the leading dimensions survive, per guard-φ
   * member.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEmbeddingDynamicSize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_embedding_dynamic_size.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, asList(new NumericDim(2), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(2), new NumericDim(2), UnresolvedDim.INSTANCE)))));
  }

  /**
   * Tier-5 generator (wala/ML#449): {@code tf.math.top_k(input, k)}. Returns a {@code (values,
   * indices)} 2-tuple. The dedicated {@link com.ibm.wala.cast.python.ml.client.TopK} generator
   * implements {@link com.ibm.wala.cast.python.ml.client.TupleElementProvider} with per-index dtype
   * precision ({@code values} inherits input dtype; {@code indices} is fixed at {@code int32}), but
   * the wrap-on-property-read dispatch in {@link
   * com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory} doesn't fire for NamedTuple-style
   * attribute access ({@code result.values} / {@code result.indices}). Until <a
   * href="https://github.com/wala/ML/issues/480">wala/ML#480</a> is fixed, both destructured
   * elements receive the aggregate union of per-element types.
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/480">wala/ML#480</a> lands, narrow the
   * assertion to {@code Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)} (precise {@code values} dtype).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTopKValues()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_top_k.py",
        "f_values",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32, TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * Counterpart of {@link #testTopKValues()} for the {@code indices} element of {@code
   * tf.math.top_k}'s tuple result. Same wala/ML#480-driven imprecision: the assertion captures the
   * observed aggregate union with a TODO to narrow once the per-index dispatch is fixed.
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/480">wala/ML#480</a> lands, narrow the
   * assertion to {@code Set.of(TENSOR_INT32_UNKNOWN_SHAPE)} (precise {@code indices} dtype).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTopKIndices()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_top_k.py",
        "f_indices",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32, TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * Tier-5 generator (wala/ML#449): {@code tf.meshgrid(*xi)}. Returns N tensors (one per input)
   * sharing the broadcast of input shapes and the first input's dtype. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Meshgrid} generator implements {@link
   * com.ibm.wala.cast.python.ml.client.TupleElementProvider}, but the XML only allocates one tuple
   * slot (field 0); the second meshgrid output ({@code Y} in the fixture) doesn't have a backing
   * alloc, so it falls through to ⊤/UNKNOWN and the aggregate union leaks the {@code
   * float32}/{@code unknown} pair. See <a href="https://github.com/wala/ML/issues/480">
   * wala/ML#480</a>.
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/480">wala/ML#480</a> lands (or the
   * meshgrid XML is updated to allocate per-input tuple slots), narrow the assertion to {@code
   * Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testMeshgrid()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_meshgrid.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32, TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Companion to {@link #testMeshgrid}: 2-parameter sink {@code f(X, Y)} so the analyzer's
   * per-parameter typing on `tf.meshgrid`'s tuple result can be observed at distinct value numbers.
   * The two parameters split the leak asymmetrically:
   *
   * <ul>
   *   <li>vn=2 ({@code X}) shows the full {@code float32}/⊤-dtype union &mdash; the meshgrid XML
   *       only allocates field-0 of the tuple, so when {@code X} aliases that slot through PA
   *       propagation it picks up both the precise float32 alloc and the ⊤ tuple-slot leak.
   *   <li>vn=3 ({@code Y}) collapses to just the precise {@code float32} &mdash; the second
   *       meshgrid output's allocation site doesn't reach this parameter through the PA graph, so
   *       the ⊤ leak doesn't surface.
   * </ul>
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/480">wala/ML#480</a> lands, narrow
   * vn=2's assertion to {@code Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)} (vn=3 is already there).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testMeshgridXY()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_meshgrid_xy.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32, TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE),
            3, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Generator-dispatch test for the 3-arg form of {@code tf.where(condition, x, y)}. The dedicated
   * {@link com.ibm.wala.cast.python.ml.client.Where} generator produces shape and dtype by unioning
   * the inferred sets over {@code x} and {@code y} (and intentionally ignoring {@code condition}'s
   * shape, per the modeling note in {@code Where}'s class Javadoc). The fixture has all three
   * operands shape {@code (3,)} float32, so the union collapses to the singleton {@code (3,)
   * float32}. Fix for the wala/ML#422 listing.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testWhere()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_where.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Sibling of {@link #testWhere()} that exercises the union-over-{@code x}-and-{@code y} path with
   * broadcast-compatible different shapes: {@code x} is {@code (3,)} float32 and {@code y} is
   * {@code (2, 3)} float32. The runtime broadcast result is {@code (2, 3)}; the static analysis
   * unions the two operand shapes to {@code {(3,), (2, 3)}}, which is sound but imprecise.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/482">wala/ML#482</a>): once broadcast-shape
   * composition lands, tighten this assertion from the union {@code {TENSOR_3_FLOAT32,
   * TENSOR_2_3_FLOAT32}} to the precise {@code TENSOR_2_3_FLOAT32}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testWhereBroadcast()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_where_broadcast.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_FLOAT32, TENSOR_2_3_FLOAT32)));
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
   * Regression guard for wala/ML#403: chained {@code x.astype(int32).astype(float32)} on an mnist
   * receiver. The first cast's result is a synthetic-method return whose PointerKey is implicit, so
   * the receiver-shape lookup for the second {@code astype} call hits the {@code
   * IllegalArgumentException}-catch fallback path in {@link AstypeOperation#getDefaultShapes},
   * which returns {@code null} (⊤) for shape while dtype still resolves to {@code float32}.
   *
   * <p>The runtime-vs-analyzer asymmetry here ({@code (60000, 28, 28) float32} at runtime vs.
   * {@code TensorType(float32, null)} from the analyzer) is the same kind of deliberate limitation
   * as {@link #testInputUnresolvableShape}: traversing implicit-PK chains across synthetic-method
   * returns is a known architectural gap (wala/ML#402 / wala/WALA#1889), and returning ⊤ rather
   * than ⊥ is the lattice-correct response in the meantime — dtype still carries through, so
   * downstream analysis isn't dropped. The test asserts the analyzer's lattice-correct output
   * rather than suppressing it as a would-be-fixed failure; if and when the implicit-PK chain
   * traversal lands, this expectation flips to {@code TENSOR_60000_28_28_FLOAT32} as part of that
   * change.
   */
  @Test
  public void testAstypeChained()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_astype_chained.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Exercise {@link NdarrayReshape#getDefaultShapes}'s SSA-substrate DU walk: resolves the {@code
   * -1} in {@code x_train.reshape([-1, 784])} by tracing the receiver back to {@code mnist.x_train}
   * ({@code (60000, 28, 28)}). Guards {@link TensorGenerator#getShapesOrSSAChain} against
   * regression.
   *
   * <p>Dtype propagation ({@code uint8}) uses the existing {@code getDTypes(builder, receiverVn)}
   * path, which resolves through the normal PA because {@code x_train}'s dtype is carried on the
   * {@link MnistInputData}-manufactured allocation.
   */
  @Test
  public void testNdarrayReshape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ndarray_reshape.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_60000_784_UINT8)));
  }

  /**
   * Guards the {@code tf.random.gamma} unresolvable-{@code shape} floor (<a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>): when the {@code shape} argument
   * is unresolvable (here from {@code json.loads}) the output rank can't be known, so the result
   * floors to ⊤ rather than throwing (which previously aborted the whole analysis). The dtype stays
   * float32.
   */
  @Test
  public void testGammaUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gamma_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Guards the {@code tf.random.poisson} unresolvable-{@code shape} floor (<a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>): same as {@link
   * #testGammaUnresolvable()}, the output rank rides on the unresolvable {@code shape}, so the
   * result floors to ⊤ rather than throwing.
   */
  @Test
  public void testPoissonUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_poisson_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Covers the all-numeric {@link TensorType#of} factories (<a
   * href="https://github.com/wala/ML/issues/594">wala/ML#594</a>): the {@code DType} and {@code
   * String} cell-type overloads map {@code int} dimensions to {@link NumericDim}s, equivalent to
   * the explicit-{@code List} construction, and compose with {@link TensorType#asSparse()}.
   */
  @Test
  public void testTensorTypeNumericFactory() {
    assertEquals(
        new TensorType(FLOAT32, asList(new NumericDim(2), new NumericDim(3))),
        TensorType.of(FLOAT32, 2, 3));
    assertEquals(new TensorType(FLOAT_32, asList(new NumericDim(3))), TensorType.of(FLOAT_32, 3));
    assertTrue(TensorType.of(FLOAT32, 2, 2).asSparse().isSparse());
  }

  /**
   * Regression guard for the "layer-output flows into script-level consumer" pattern: a downstream
   * function whose tensor parameter comes from a layer call's output. Companion to {@link
   * #testModelCallConsume()}, which calls {@code consume(x)} <em>inside</em> {@code
   * SequentialModel.__call__}; this fixture calls {@code consume(pred)} at script-level after the
   * layer call — the same surface shape, but a different caller. The existing {@code
   * DenseCall.getDefaultShapes} SSA-chain fallback recovers the result type when the direct PTS
   * walk doesn't carry the synthetic {@code <new>} alloc through.
   */
  @Test
  public void testLayerOutputParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_layer_output_param.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_1_10_FLOAT32)));
  }

  /**
   * Variant of {@link #testLayerOutputParam()} that interposes a user-defined {@code
   * tf.keras.Model} subclass between the {@code Dense} layers and the script-level consumer —
   * exercises the same fallback through one extra level of call indirection.
   */
  @Test
  public void testLayerOutputParamViaModel()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_layer_output_param_via_model.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_10_FLOAT32)));
  }
}
