package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A submodule of a package with no {@code __init__.py}, an implicit namespace package, is imported
 * through the package as Python 3 imports it (<a
 * href="https://github.com/wala/ML/issues/1177">wala/ML#1177</a>). The import of such a package
 * bound nothing, so every submodule read through it was empty and a call through one reached no
 * callee.
 */
public class TestNamespacePackages extends AbstractTensorTest {

  private static final String[] FILES = {
    "namespace_package_proj/N/B.py",
    "namespace_package_proj/N/D/E.py",
    "namespace_package_proj/N/P/__init__.py",
    "namespace_package_proj/N/P/Q.py",
    "namespace_package_proj/A.py"
  };

  private static final String PROJECT = "namespace_package_proj";

  /** A module imported from the namespace package, {@code from N import B}. */
  @Test
  public void testFromImport() throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "N/B.py", "f", PROJECT, 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** A module of a namespace package nested in another, {@code import N.D.E}. */
  @Test
  public void testNestedNamespacePackage()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "N/D/E.py", "g", PROJECT, 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** A module of a regular package inside the namespace package, {@code from N.P import Q}. */
  @Test
  public void testRegularPackageInside()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "N/P/Q.py", "h", PROJECT, 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * A loss constructed through {@code from custom import callback} and given to {@code compile}
   * receives the labels and the predictions where the model's training step calls {@code
   * self.loss}.
   */
  @Test
  public void testLossThroughNamespacePackage()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {
          "namespace_loss_proj/custom/callback.py",
          "namespace_loss_proj/model.py",
          "namespace_loss_proj/main.py"
        },
        "custom/callback.py",
        "TransformerLoss.call",
        "namespace_loss_proj",
        2,
        3,
        Map.of(3, Set.of(TENSOR_4_INT64), 4, Set.of(TENSOR_4_3_FLOAT32)));
  }

  private static final String[] SHADOW_FILES = {
    "namespace_shadow_proj/tensorflow/example.py", "namespace_shadow_proj/main.py"
  };

  /**
   * A project directory named like a library a summary models, without an {@code __init__.py},
   * leaves {@code from tensorflow import ones} the library's, as Python's path finder does.
   */
  @Test
  public void testModeledLibraryNotShadowed()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        SHADOW_FILES,
        "main.py",
        "consume",
        "namespace_shadow_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** The same, for {@code import tensorflow as tf} and {@code tf.ones}. */
  @Test
  public void testModeledLibraryAttributeNotShadowed()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        SHADOW_FILES,
        "main.py",
        "consume_attribute",
        "namespace_shadow_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }
}
