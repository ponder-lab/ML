package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A submodule imported under an alias, {@code from pkg import mod as alias}, is the same module as
 * {@code from pkg import mod} (<a href="https://github.com/wala/ML/issues/1017">wala/ML#1017</a>):
 * a call through the alias reaches the submodule's function and class. The in-scope from-import
 * declared the imported name rather than the alias, so the alias was never bound and a program that
 * imports its own modules with aliases never constructed what those modules define.
 */
public class TestAliasedSubmoduleImport extends AbstractTensorTest {

  private static final String[] FILES = {
    "alias_import_proj/pkg/__init__.py",
    "alias_import_proj/pkg/mod.py",
    "alias_import_proj/shadow/__init__.py",
    "alias_import_proj/shadow/mod.py",
    "alias_import_proj/main.py",
    "alias_import_proj/main2.py"
  };

  private static final String PROJECT = "alias_import_proj";

  /** The un-aliased import, the control. */
  @Test
  public void testDirectImport() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILES,
        "main.py",
        "consume_direct",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** A function reached through the alias. */
  @Test
  public void testAliasedFunction() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILES,
        "main.py",
        "consume_alias",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** A class constructed and called through the alias. */
  @Test
  public void testAliasedClass() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILES,
        "main.py",
        "consume_alias_class",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 5, 4))));
  }

  /**
   * A class whose base is written through the un-aliased submodule, {@code class Child(mod.Base)}:
   * the base's initializer runs through {@code super().__init__()} and the field it sets is read on
   * the child.
   */
  @Test
  public void testBaseThroughSubmodule()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILES,
        "main.py",
        "consume_base_field",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /** The same base written through the alias, {@code class AliasChild(alias.Base)}. */
  @Test
  public void testBaseThroughAlias() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILES,
        "main.py",
        "consume_alias_base_field",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * A package whose initialization module binds a name a submodule also has: Python's {@code from
   * shadow import mod} takes the package's attribute and never loads the submodule, so the call
   * reaches the attribute's method, (7, 7), not the submodule's, (9, 9). The loader binds the name
   * to the submodule whenever a file of that name exists, so the parameter reads (9, 9).
   *
   * <p>TODO: Flip to a plain {@code @Test} when wala/ML#1019 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testPackageAttributeShadowsSubmodule()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILES,
        "main2.py",
        "consume_shadow_direct",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 7, 7))));
  }
}
