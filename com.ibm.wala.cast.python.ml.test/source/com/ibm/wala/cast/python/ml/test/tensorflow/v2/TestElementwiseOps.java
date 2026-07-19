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
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_RAGGED_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_RAGGED_RAGGED_NONE_STRING;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_INT32_UNKNOWN_SHAPE;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_32_FLOAT32;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
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
}
