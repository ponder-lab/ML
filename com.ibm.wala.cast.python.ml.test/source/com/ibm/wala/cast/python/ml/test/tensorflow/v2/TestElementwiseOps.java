package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_4_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_RAGGED_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_5_COMPLEX64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_5_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_RAGGED_RAGGED_NONE_STRING;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_INT32_UNKNOWN_SHAPE;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_32_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.UINT_8;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.AppliedDTypeCoercion;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of element-wise binary-operator inference (the {@code testAdd*} family), carved from the
 * {@code TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestElementwiseOps extends AbstractTensorTest {

  @Test
  public void testAdd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
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

  @Test
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
        Map.of(2, Set.of(TENSOR_3_4_INT32.asSparse()), 3, Set.of(TENSOR_3_4_INT32.asSparse())));
  }

  @Test
  public void testAdd29()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add29.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_4_INT32.asSparse()), 3, Set.of(TENSOR_3_4_INT32.asSparse())));
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
        Map.of(2, Set.of(TENSOR_3_4_INT32.asSparse()), 3, Set.of(TENSOR_3_4_INT32.asSparse())));
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
        Map.of(2, Set.of(TENSOR_3_4_INT32.asSparse()), 3, Set.of(TENSOR_3_4_INT32.asSparse())));
  }

  @Test
  public void testAdd64()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add64.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_4_INT32.asSparse()), 3, Set.of(TENSOR_3_4_INT32.asSparse())));
  }

  @Test
  public void testAdd65()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add65.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_4_INT32.asSparse()), 3, Set.of(TENSOR_3_4_INT32.asSparse())));
  }

  @Test
  public void testAdd66()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add66.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_5_INT32), 3, Set.of(TENSOR_1_5_INT32)));
  }

  @Test
  public void testAdd67()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add67.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_5_INT32), 3, Set.of(TENSOR_1_5_INT32)));
  }

  @Test
  public void testAdd68()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add68.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_5_INT32), 3, Set.of(TENSOR_1_5_INT32)));
  }

  @Test
  public void testAdd69()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add69.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_5_INT32), 3, Set.of(TENSOR_1_5_INT32)));
  }

  @Test
  public void testAdd70()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add70.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_5_INT32), 3, Set.of(TENSOR_1_5_INT32)));
  }

  @Test
  public void testAdd71()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add71.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_5_INT32), 3, Set.of(TENSOR_1_5_INT32)));
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
            2,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING),
            3,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING)));
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
            2,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING),
            3,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING)));
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
            2,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING),
            3,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING)));
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
            2,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING),
            3,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING)));
  }

  @Test
  public void testAdd76()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add76.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_INT32), 3, Set.of(TENSOR_3_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testAdd77()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add77.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_INT32), 3, Set.of(TENSOR_3_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testAdd78()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add78.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_INT32), 3, Set.of(TENSOR_3_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testAdd79()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add79.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_INT32), 3, Set.of(TENSOR_3_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testAdd80()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add80.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_INT32), 3, Set.of(TENSOR_3_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testAdd81()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add81.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_INT32), 3, Set.of(TENSOR_3_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testAdd82()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add82.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_INT32), 3, Set.of(TENSOR_3_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testAdd83()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add83.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_INT32), 3, Set.of(TENSOR_3_RAGGED_RAGGED_INT32)));
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
            2,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING),
            3,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING)));
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
            2,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING),
            3,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING)));
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
            2,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING),
            3,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING)));
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
            2,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING),
            3,
            Set.of(TENSOR_4_RAGGED_RAGGED_NONE_STRING)));
  }

  @Test
  public void testAdd88()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add88.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_RAGGED_INT32), 3, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testAdd89()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add89.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_RAGGED_INT32), 3, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testAdd90()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add90.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_RAGGED_INT32), 3, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testAdd91()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add91.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_RAGGED_INT32), 3, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testAdd92()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add92.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_RAGGED_INT32), 3, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testAdd93()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add93.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_RAGGED_INT32), 3, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testAdd94()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add94.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_RAGGED_INT32), 3, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testAdd95()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add95.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_RAGGED_INT32), 3, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testAdd96()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add96.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_5_RAGGED_INT32), 3, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testAdd97()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add97.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32), 3, Set.of(TENSOR_4_RAGGED_INT32)));
  }

  @Test
  public void testAdd98()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add98.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32), 3, Set.of(TENSOR_4_RAGGED_INT32)));
  }

  @Test
  public void testAdd99()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add99.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32), 3, Set.of(TENSOR_4_RAGGED_INT32)));
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
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32.asSparse()), 3, Set.of(TENSOR_2_3_FLOAT32.asSparse())));
  }

  @Test
  public void testAdd112()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add112.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32.asSparse()), 3, Set.of(TENSOR_2_3_FLOAT32.asSparse())));
  }

  @Test
  public void testAdd113()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add113.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32.asSparse()), 3, Set.of(TENSOR_2_3_FLOAT32.asSparse())));
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
   * Tests that under per-context shape unions the analysis returns the broadcastable cross-pairs
   * and silently discards the non-broadcastable ones (wala/ML#462).
   *
   * <p>In {@code tf2_test_add117.py}, the variable {@code a} can be either 1 or 3.
   *
   * <ul>
   *   <li>If {@code a=1}, the addition is {@code [1, 2] + [2, 2]}, which is broadcastable to {@code
   *       [2, 2]}.
   *   <li>If {@code a=3}, the addition is {@code [3, 2] + [2, 2]}, which is NOT broadcastable.
   * </ul>
   *
   * The analysis retains the broadcastable result ({@code [2, 2]}) and discards the
   * non-broadcastable cross-pair as analysis-level imprecision rather than a runtime error &mdash;
   * the cross-pair would never co-occur at runtime under matched contexts. When <em>every</em> pair
   * is non-broadcastable, the result shape instead degrades to ⊤ (<a
   * href="https://github.com/wala/ML/issues/583">wala/ML#583</a>).
   *
   * @see #testAdd117a()
   */
  @Test
  public void testAdd117()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add117.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32, TENSOR_3_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
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

  /**
   * Probes <a href="https://github.com/wala/ML/issues/743">wala/ML#743</a>: an unpassed parameter
   * default (the float scalar {@code eps}) must materialize in the pointer analysis for the
   * elementwise addition to resolve. The result keeps the tensor operand's shape (scalar broadcast)
   * and dtype.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testParamDefault()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_param_default.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * The shape-and-dtype-preserving {@code tf.image} augmentation chain (<a
   * href="https://github.com/wala/ML/issues/792">wala/ML#792</a>): {@code adjust_brightness},
   * {@code random_flip_left_right}, {@code adjust_contrast}, {@code adjust_saturation}, and {@code
   * adjust_hue} each pass the image argument's shape and dtype through, so the {@code (4, 5, 3)}
   * uint8 constant survives all five hops.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testImageAugmentationChain()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_image_augmentation.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(UINT_8, 4, 5, 3))));
  }

  /**
   * The cropped companion of {@link #testImageAugmentationChain()}: the image passes through a
   * {@code tf.slice} hop into {@code adjust_brightness}, matching the corpus augmentation chain's
   * crop-then-adjust structure. With constant extents the sliced shape computes exactly
   * (wala/ML#569); the corpus's dynamically sized crop rides the same dtype path with the shape
   * axis at ⊤ instead.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testImageAugmentationCroppedChain()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_image_augmentation.py",
        "consume2",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(UINT_8, 2, 2, 3))));
  }

  /**
   * Test a {@code W * x + b} chain whose {@code W} and {@code b} are scalar {@code tf.Variable}s,
   * against the runtime truth of {@code float32 (3,)}. Mirrors the TensorFlow-Examples
   * linear-regression rows whose concrete member degraded to ⊤ at 0.52.73, when the wala/ML#796
   * null-skip removed the fabricated scalar member the broadcast had been riding.
   *
   * <p>The chain used to floor to ⊤ because {@code W}'s initializer is {@code rng.randn()} and
   * {@code np.random} was unmodeled, so the operand was ⊤ rather than the rank-0 tensor it is. With
   * the initializer resolving (<a href="https://github.com/wala/ML/issues/827">wala/ML#827</a>) the
   * operand is scalar outright, and the elementwise composition never reaches the one-sided
   * broadcast rule that <a href="https://github.com/wala/ML/issues/804">wala/ML#804</a> reports on.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testScalarVariableBroadcast()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_scalar_variable_broadcast.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pins the dtype of an integral NumPy array divided by a Python float (wala/ML#814). NumPy
   * promotes to {@code float64}, because a Python float is a double. The previous rule answered
   * {@code float32} for every float-literal operand, citing this very idiom.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNumpyIntegerDividedByFloatLiteral()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_numpy_float_division.py",
        "consume_int_scaled",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_5_FLOAT64)));
  }

  /**
   * Companion to {@link #testNumpyIntegerDividedByFloatLiteral} (wala/ML#814): a {@code float32}
   * array keeps {@code float32} under the same operation, so the rule is not "a float literal makes
   * the result {@code float64}" either. The operand's own dtype decides.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNumpyFloat32DividedByFloatLiteral()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_numpy_float_division.py",
        "consume_float_scaled",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_5_FLOAT32)));
  }

  /**
   * Companion to {@link #testNumpyIntegerDividedByFloatLiteral} for a complex operand
   * (wala/ML#814). A complex array keeps its own dtype, so the rule is that an integral operand
   * widens to {@code float64} while every other numeric operand keeps itself. Testing
   * floating-point-ness rather than numeric-ness here would send complex operands to the {@code
   * float32} fallback and lose the imaginary part.
   *
   * <p>TODO: Flip to a plain {@code @Test} when <a
   * href="https://github.com/wala/ML/issues/816">wala/ML#816</a> lands. The promotion rule here is
   * already correct for complex operands; the operand never arrives as one, because {@code
   * np.zeros(..., dtype=np.complex64)} is itself typed {@code float64}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test(expected = AssertionError.class)
  public void testNumpyComplexDividedByFloatLiteral()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_numpy_float_division.py",
        "consume_complex_scaled",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_5_COMPLEX64)));
  }

  /**
   * The eager-coercion parameter dtype of <a
   * href="https://github.com/wala/ML/issues/828">wala/ML#828</a>: {@code W * x} over a {@code
   * float32} variable makes the parameter's eager-effective dtype {@code float32}, though its
   * caller feeds a {@code float64} array. Naming the fed dtype in a signature is what turns a
   * working eager call into a failing traced one.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEagerCoercionDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eager_coercion_dtype.py", "coerced", 1, 3, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * The disagreement arm of {@link #testEagerCoercionDType()}: two consumers impose different
   * dtypes on the same parameter, each succeeding on its own, so the parameter is reported as the
   * union over the imposed dtypes — the body genuinely computes with {@code float32} at one op and
   * {@code float64} at the other — rather than as the fed dtype, which is not a claim the analysis
   * believes there (wala/ML#829).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEagerCoercionDTypeDisagreeing()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_eager_coercion_dtype.py",
        "disagreeing",
        1,
        5,
        Map.of(2, Set.of(TENSOR_3_FLOAT32, TENSOR_3_FLOAT64)));
  }

  /**
   * The unaccounted arm of {@link #testEagerCoercionDType()}: one consumer imposes {@code float32},
   * but the other ({@code tf.tensordot}) takes a second operand and declares no coercion, so the
   * collection is incomplete and the parameter keeps the dtype it is fed — a union here would
   * attach a confident-looking answer to ignorance (wala/ML#829).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEagerCoercionDTypeUnaccounted()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_eager_coercion_dtype.py",
        "unaccounted",
        1,
        5,
        Map.of(2, Set.of(TENSOR_3_FLOAT64)));
  }

  /**
   * The keyword form of {@link #testEagerCoercionDTypeUnaccounted()}: the undeclared consumer's
   * second operand arrives by keyword ({@code tf.tensordot(x, b=V, axes=0)}), so a positional-only
   * account of the call's arity would let it slip past the unaccounted decline and pin {@code
   * float32} from the remaining consumer alone (wala/ML#829).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEagerCoercionDTypeKeywordUnaccounted()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_eager_coercion_dtype.py",
        "keyword_unaccounted",
        1,
        5,
        Map.of(2, Set.of(TENSOR_3_FLOAT64)));
  }

  /**
   * The applied coercions are exposed with their FED side (<a
   * href="https://github.com/wala/ML/issues/838">wala/ML#838</a>): a client converting a function
   * without writing a declaration decides safety by whether the coercion CHANGED anything, which
   * the imposed-only report cannot answer. The {@code coerced} arm changed (fed {@code float64},
   * imposed {@code float32}); the {@code agreeing} arm did not (a complete fed side equal to the
   * imposition); the {@code disagreeing} arm reads changed whatever the fed side, since a union
   * means some consumer computes at a dtype no single materialization satisfies; the {@code
   * chained} inner parameter reads UNRESOLVED, since its inflow carries the upstream parameter's
   * IMPOSED dtype where the runtime forwards the original value; the {@code shaped} parameter reads
   * UNRESOLVED, since the {@code set_shape} pin owns its inflow edges and the coercion transfer
   * observes no feed; declined parameters carry no record at all.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testAppliedDTypeCoercionExposure()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonTensorAnalysisEngine engine = makeEngine(emptyList(), "tf2_test_eager_coercion_dtype.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    Map<String, List<AppliedDTypeCoercion>> byFunction = new HashMap<>();
    for (Map.Entry<PointerKey, AppliedDTypeCoercion> entry :
        analysis.getAppliedDTypeCoercions().entrySet()) {
      if (!(entry.getKey() instanceof LocalPointerKey lpk)) continue;
      // The declaring class NAME is the full per-file function identity for Python functions
      // (Lscript <file>/<function>), so same-named functions in other files cannot merge; the
      // List (rather than a set) keeps value-equal records from distinct parameters or contexts
      // countable, so the cardinality guard below cannot pass vacuously.
      String function =
          lpk.getNode().getMethod().getDeclaringClass().getName().getClassName().toString();
      byFunction.computeIfAbsent(function, k -> new ArrayList<>()).add(entry.getValue());
    }

    AppliedDTypeCoercion coerced = only(byFunction, "coerced");
    assertEquals(EnumSet.of(FLOAT64), coerced.fed());
    assertEquals(EnumSet.of(FLOAT32), coerced.imposed());
    assertEquals(AppliedDTypeCoercion.Resolution.CHANGED, coerced.resolution());
    assertTrue("A replaced fed dtype reads changed.", coerced.changed());

    AppliedDTypeCoercion agreeing = only(byFunction, "agreeing");
    assertEquals(EnumSet.of(FLOAT32), agreeing.fed());
    assertEquals(EnumSet.of(FLOAT32), agreeing.imposed());
    assertEquals(AppliedDTypeCoercion.Resolution.UNCHANGED, agreeing.resolution());
    assertTrue("A complete equal imposition reads unchanged.", !agreeing.changed());

    AppliedDTypeCoercion disagreeing = only(byFunction, "disagreeing");
    assertEquals(EnumSet.of(FLOAT64), disagreeing.fed());
    assertEquals(EnumSet.of(FLOAT32, FLOAT64), disagreeing.imposed());
    assertEquals(AppliedDTypeCoercion.Resolution.CHANGED, disagreeing.resolution());

    // The chained inner parameter: the analysis-side inflow is the upstream parameter's coerced
    // state, so the record must degrade rather than read a false unchanged.
    AppliedDTypeCoercion chainedInner = only(byFunction, "chained_inner");
    assertEquals(AppliedDTypeCoercion.Resolution.UNRESOLVED, chainedInner.resolution());

    // The `set_shape` pin owns the inflow EDGES, but the predecessors' states are the runtime
    // feed regardless of which transfer the analysis used: a resolved float32 tensor feed under
    // an equal imposition is safe.
    AppliedDTypeCoercion shaped = only(byFunction, "shaped");
    assertEquals(EnumSet.of(FLOAT32), shaped.fed());
    assertEquals(AppliedDTypeCoercion.Resolution.UNCHANGED, shaped.resolution());

    assertTrue("A declined parameter carries no record.", !byFunction.containsKey("unaccounted"));
    assertTrue(
        "A circularly declined parameter carries no record.", !byFunction.containsKey("circular"));

    // The record's classification, pinned directly: an empty fed side and an incomplete fed side
    // are both UNRESOLVED, never vacuously unchanged (absence is not evidence of absence), and
    // the two-valued convenience folds UNRESOLVED into `changed` (wrong-but-safe for a
    // safety-deciding client).
    assertEquals(
        AppliedDTypeCoercion.Resolution.UNRESOLVED,
        new AppliedDTypeCoercion(EnumSet.noneOf(DType.class), EnumSet.of(FLOAT32), true)
            .resolution());
    AppliedDTypeCoercion incomplete =
        new AppliedDTypeCoercion(EnumSet.of(FLOAT32), EnumSet.of(FLOAT32), false);
    assertEquals(AppliedDTypeCoercion.Resolution.UNRESOLVED, incomplete.resolution());
    assertTrue("An unresolved record folds into changed.", incomplete.changed());
  }

  /**
   * The single coercion record of one fixture function, asserting the cardinality so a second
   * recorded parameter or context cannot silently make the assertions test an arbitrary survivor.
   *
   * @param byFunction The records grouped by function name.
   * @param function The fixture function.
   * @return Its single record.
   */
  private static AppliedDTypeCoercion only(
      Map<String, List<AppliedDTypeCoercion>> byFunction, String function) {
    List<AppliedDTypeCoercion> records = byFunction.get(function);
    assertNotNull("The " + function + " arm must carry a record.", records);
    assertEquals("One record for " + function + ".", 1, records.size());
    return records.get(0);
  }

  /**
   * The declared-but-unresolved arm of {@link #testEagerCoercionDTypeUnaccounted()}: {@code x * y}
   * is a declared consumer, but each operand's partner is itself a parameter, so no imposed dtype
   * can be established there without circularity — the account is incomplete and both parameters
   * keep the dtype they are fed, rather than {@code x} pinning to the {@code float32} its other
   * consumer imposes (wala/ML#829).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEagerCoercionDTypeCircular()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_eager_coercion_dtype.py",
        "circular",
        2,
        5,
        Map.of(2, Set.of(TENSOR_3_FLOAT64), 3, Set.of(TENSOR_3_FLOAT64)));
  }

  /**
   * The no-op arm of {@link #testEagerCoercionDType()}: the argument already carries the dtype its
   * consumer imposes, so the coercion is recorded but changes nothing.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEagerCoercionDTypeAgreeing()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eager_coercion_dtype.py", "agreeing", 1, 3, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * The parameter half of {@link #testScalarVariableBroadcast()}, and the guard <a
   * href="https://github.com/wala/ML/issues/828">wala/ML#828</a> asks for: the same {@code W * x +
   * b} body, asserted on the parameter rather than on the sink. Its caller feeds a {@code float64}
   * array, but the body computes in the variable's {@code float32}, and a signature written from
   * the fed dtype is what makes the traced call raise where the eager one runs.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testScalarVariableBroadcastParameter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_scalar_variable_broadcast.py",
        "linear_regression",
        1,
        4,
        Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }
}
