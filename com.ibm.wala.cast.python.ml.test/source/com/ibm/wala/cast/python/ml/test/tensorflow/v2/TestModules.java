package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_FLOAT32;
import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of module and import binding mechanics ({@code Module*}/{@code Import*}), carved from the
 * {@link TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestModules extends AbstractTensorTest {

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
}
