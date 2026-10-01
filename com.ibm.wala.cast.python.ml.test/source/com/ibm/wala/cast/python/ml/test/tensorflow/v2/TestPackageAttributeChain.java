package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a call written through a package's attribute chain reaches the function or class the
 * package's initialization scripts bind (<a href="https://github.com/wala/ML/issues/210">wala/ML
 * #210</a>). Package {@code pkg} binds its subpackage with a relative import and package {@code
 * qkg} with an absolute one.
 */
public class TestPackageAttributeChain extends AbstractTensorTest {

  private static final String[] FILES = {
    "pkgchain_proj/pkg/__init__.py",
    "pkgchain_proj/pkg/sub/__init__.py",
    "pkgchain_proj/pkg/sub/mod.py",
    "pkgchain_proj/qkg/__init__.py",
    "pkgchain_proj/qkg/sub/__init__.py",
    "pkgchain_proj/qkg/sub/mod.py",
    "pkgchain_proj/main.py",
    "pkgchain_proj/main2.py",
    "pkgchain_proj/main3.py",
    "pkgchain_proj/main4.py"
  };

  private void check(String file, String function, int param, int columns)
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILES,
        file,
        function,
        "pkgchain_proj",
        1,
        2,
        Map.of(param, Set.of(TensorType.of(FLOAT_32, 3, columns))));
  }

  /** {@code import pkg; pkg.sub.f(x)}: a function a subpackage re-exports. */
  @Test
  public void testFunctionThroughChain()
      throws ClassHierarchyException, CancelException, IOException {
    check("pkg/sub/mod.py", "f", 2, 4);
  }

  /** {@code from pkg.sub import g; g(x)}: the name imported directly. */
  @Test
  public void testFunctionImportedByName()
      throws ClassHierarchyException, CancelException, IOException {
    check("pkg/sub/mod.py", "g", 2, 5);
  }

  /** {@code import pkg; pkg.sub.mod.h(x)}: a function read through its module. */
  @Test
  public void testFunctionThroughModule()
      throws ClassHierarchyException, CancelException, IOException {
    check("pkg/sub/mod.py", "h", 2, 6);
  }

  /** {@code import pkg; pkg.sub.Scale()(x)}: a class a subpackage re-exports. */
  @Test
  public void testClassThroughChain() throws ClassHierarchyException, CancelException, IOException {
    check("pkg/sub/mod.py", "Scale.__call__", 3, 7);
  }

  /** {@code from pkg import sub; sub.f2(x)}: the subpackage imported by name. */
  @Test
  public void testFromPackageImportSubpackage()
      throws ClassHierarchyException, CancelException, IOException {
    check("pkg/sub/mod.py", "f2", 2, 8);
  }

  /** {@code import pkg.sub; pkg.sub.f3(x)}: the subpackage imported by its dotted name. */
  @Test
  public void testImportSubpackageDotted()
      throws ClassHierarchyException, CancelException, IOException {
    check("pkg/sub/mod.py", "f3", 2, 9);
  }

  /** {@code import pkg.sub.mod as m; m.f4(x)}: a module imported under an alias. */
  @Test
  public void testImportModuleAs() throws ClassHierarchyException, CancelException, IOException {
    check("pkg/sub/mod.py", "f4", 2, 10);
  }

  /** {@code s = pkg.sub; s.f5(x)}: the subpackage read into a local. */
  @Test
  public void testChainBoundToLocal() throws ClassHierarchyException, CancelException, IOException {
    check("pkg/sub/mod.py", "f5", 2, 11);
  }

  /** {@code import pkg as p; p.sub.f6(x)}: the package imported under an alias. */
  @Test
  public void testImportPackageAs() throws ClassHierarchyException, CancelException, IOException {
    check("pkg/sub/mod.py", "f6", 2, 14);
  }

  /** {@code import pkg.sub as s; s.f7(x)}: the subpackage imported under an alias. */
  @Test
  public void testImportSubpackageAs()
      throws ClassHierarchyException, CancelException, IOException {
    check("pkg/sub/mod.py", "f7", 2, 15);
  }

  /** {@code import qkg; qkg.sub.k(x)}: initialization scripts that import absolutely. */
  @Test
  public void testAbsoluteImportChain()
      throws ClassHierarchyException, CancelException, IOException {
    check("qkg/sub/mod.py", "k", 2, 12);
  }

  /** {@code import qkg; qkg.sub.Scaler()(x)}: as above, for a class. */
  @Test
  public void testAbsoluteImportChainClass()
      throws ClassHierarchyException, CancelException, IOException {
    check("qkg/sub/mod.py", "Scaler.__call__", 3, 13);
  }
}
