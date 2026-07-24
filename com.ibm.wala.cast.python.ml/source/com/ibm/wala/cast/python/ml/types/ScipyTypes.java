package com.ibm.wala.cast.python.ml.types;

import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;

/**
 * Types found in the SciPy library, as modeled in {@code scipy.xml}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ScipyTypes extends PythonTypes {

  public static final TypeReference SCIPY_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Lscipy"));

  public static final TypeReference SPARSE_MATRIX_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Lscipy/sparse/spmatrix"));

  /** https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html */
  public static final MethodReference DIAGS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lscipy/sparse/diags")),
          AstMethodReference.fnSelector);

  /** https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.dot.html */
  public static final MethodReference SPARSE_MATRIX_DOT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lscipy/sparse/spmatrix/dot")),
          AstMethodReference.fnSelector);

  private ScipyTypes() {}
}
