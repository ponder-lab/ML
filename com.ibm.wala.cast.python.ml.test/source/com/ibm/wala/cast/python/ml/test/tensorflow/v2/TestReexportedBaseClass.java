package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A class whose base is a class a package re-exports, {@code class C(pkg.Base)} or {@code from pkg
 * import Base; class C(Base)} where {@code pkg/__init__.py} imports {@code Base} from the module
 * that defines it, inherits that base (<a href="https://github.com/wala/ML/issues/1042">
 * wala/ML#1042</a>). The base was looked up only among the classes the named module defines, and a
 * package's initialization module defines none of the classes it re-exports, so the class fell to
 * {@code object}: the base's initializer never ran on its instances, the fields it sets were never
 * written, and a call on an instance did not dispatch to the inherited layer {@code call}.
 */
public class TestReexportedBaseClass extends AbstractTensorTest {

  private static final String[] FILES = {
    "reexport_base_proj/pkg/__init__.py",
    "reexport_base_proj/pkg/inputters/__init__.py",
    "reexport_base_proj/pkg/inputters/inputter.py",
    "reexport_base_proj/pkg/models/__init__.py",
    "reexport_base_proj/pkg/models/s2s.py",
    "reexport_base_proj/main.py"
  };

  private static final String PROJECT = "reexport_base_proj";

  private static final Map<Integer, Set<TensorType>> FLOAT_2_3 =
      Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3)));

  /** The base imported from the module that defines it, the control. */
  @Test
  public void testBaseFromDefiningModule()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_defining_module", PROJECT, 1, 1, FLOAT_2_3);
  }

  /**
   * The base written as an attribute of the re-exporting package, {@code class
   * C(inputters.ExampleInputter)}: the field the base's initializer sets is read on the subclass's
   * instance.
   */
  @Test
  public void testBaseThroughPackageAttribute()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_package_attribute", PROJECT, 1, 1, FLOAT_2_3);
  }

  /** The base imported by name from the re-exporting package, {@code from pkg.inputters import}. */
  @Test
  public void testBaseThroughPackageName()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_package_name", PROJECT, 1, 1, FLOAT_2_3);
  }

  /** The base re-exported by one package from another that re-exports it, two hops. */
  @Test
  public void testBaseThroughTwoPackages()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_two_packages", PROJECT, 1, 1, FLOAT_2_3);
  }

  /**
   * A call on an instance of the subclass dispatches to the layer {@code call} it inherits through
   * the re-exported base.
   */
  @Test
  public void testLayerCallThroughPackageAttribute()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_layer_call", PROJECT, 1, 1, FLOAT_2_3);
  }

  /**
   * The base re-exported by a relative import in the package's initialization module, {@code from
   * .inputter import Inputter}.
   */
  @Test
  public void testBaseThroughRelativeReexport()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILES, "main.py", "consume_relative_reexport", PROJECT, 1, 1, FLOAT_2_3);
  }
}
