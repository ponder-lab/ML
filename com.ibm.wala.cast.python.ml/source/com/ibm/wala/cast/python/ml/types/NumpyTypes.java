package com.ibm.wala.cast.python.ml.types;

import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import java.util.Locale;
import java.util.Map;

/**
 * Types found in the NumPy library.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NumpyTypes extends PythonTypes {

  /** Defined data types used in NumPy. */
  public enum DType {
    FLOAT32(true, true, 32),
    FLOAT64(true, true, 64),
    INT32(true, false, 32),
    INT64(true, false, 64),
    UINT8(true, false, 8),
    STRING(false, false, 0),
    UNKNOWN(false, false, 0);

    private final boolean numeric;
    private final boolean floatingPoint;
    private final int precision;

    DType(boolean numeric, boolean floatingPoint, int precision) {
      this.numeric = numeric;
      this.floatingPoint = floatingPoint;
      this.precision = precision;
    }

    public boolean canConvertTo(DType other) {
      if (other == null) return false;
      if (!this.numeric || !other.numeric) return this == other;
      if (this.floatingPoint && !other.floatingPoint) return false;
      return this.precision <= other.precision;
    }
  }

  public static final String NUMPY = "numpy";

  public static final TypeReference NUMPY_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Lnumpy"));

  public static final TypeReference NDARRAY_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Lnumpy/ndarray"));

  public static final TypeReference D_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Lnumpy/dtype"));

  /** https://numpy.org/doc/stable/reference/generated/numpy.array.html */
  public static final MethodReference ARRAY =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/array")),
          AstMethodReference.fnSelector);

  private static final String ARRAY_SIGNATURE = "numpy.array()";

  /** https://numpy.org/doc/stable/reference/generated/numpy.zeros.html */
  public static final MethodReference ZEROS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/zeros")),
          AstMethodReference.fnSelector);

  private static final String ZEROS_SIGNATURE = "numpy.zeros()";

  /** https://numpy.org/doc/stable/reference/generated/numpy.ones.html */
  public static final MethodReference ONES =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/ones")),
          AstMethodReference.fnSelector);

  private static final String ONES_SIGNATURE = "numpy.ones()";

  /** https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html */
  public static final MethodReference NDARRAY_CONSTRUCTOR =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/ndarray_constructor")),
          AstMethodReference.fnSelector);

  private static final String NDARRAY_CONSTRUCTOR_SIGNATURE = "numpy.ndarray()";

  /** https://numpy.org/doc/stable/reference/generated/numpy.eye.html */
  public static final MethodReference EYE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/eye")),
          AstMethodReference.fnSelector);

  private static final String EYE_SIGNATURE = "numpy.eye()";

  /** https://numpy.org/doc/stable/reference/generated/numpy.unique.html */
  public static final MethodReference UNIQUE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/unique")),
          AstMethodReference.fnSelector);

  private static final String UNIQUE_SIGNATURE = "numpy.unique()";

  /** Slot 0 of {@link #UNIQUE}: the unique values, dtype preserved from the input. wala/ML#799. */
  public static final MethodReference UNIQUE_VALUES =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/unique_values")),
          AstMethodReference.fnSelector);

  private static final String UNIQUE_VALUES_SIGNATURE = "numpy.unique() values";

  /** Slots 1-3 of {@link #UNIQUE}: index, inverse, and counts, always int64. wala/ML#799. */
  public static final MethodReference UNIQUE_INDICES =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/unique_indices")),
          AstMethodReference.fnSelector);

  private static final String UNIQUE_INDICES_SIGNATURE = "numpy.unique() indices";

  /** https://numpy.org/doc/stable/reference/generated/numpy.reshape.html */
  public static final MethodReference RESHAPE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/reshape")),
          AstMethodReference.fnSelector);

  private static final String RESHAPE_SIGNATURE = "numpy.reshape()";

  /** https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html */
  public static final MethodReference RESHAPE_METHOD =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/ndarray/reshape")),
          AstMethodReference.fnSelector);

  private static final String RESHAPE_METHOD_SIGNATURE = "numpy.ndarray.reshape()";

  /** Method name used in {@code numpy.xml} for {@link #ASTYPE}. */
  public static final String ASTYPE_METHOD_NAME = "astype";

  /** https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html */
  public static final MethodReference ASTYPE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/ndarray/astype")),
          AstMethodReference.fnSelector);

  private static final String ASTYPE_SIGNATURE = "numpy.ndarray.astype()";

  /** Method name used in {@code numpy.xml} for {@link #TOLIST}. */
  public static final String TOLIST_METHOD_NAME = "tolist";

  /** https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html */
  public static final MethodReference TOLIST =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/ndarray/tolist")),
          AstMethodReference.fnSelector);

  private static final String TOLIST_SIGNATURE = "numpy.ndarray.tolist()";

  /**
   * The allocation-only marker class for {@link #TOLIST} results: a distinct class rather than
   * {@code Lnumpy/ndarray} so producer delegation dispatches {@code TolistOperation} without
   * conflating the value with a real ndarray. wala/ML#796.
   */
  public static final TypeReference TOLIST_RESULT =
      TypeReference.findOrCreate(
          PythonTypes.pythonLoader, TypeName.string2TypeName("Lnumpy/ndarray/tolist_result"));

  /** A mapping from a {@link TypeReference} to its associated NumPy signature. */
  public static final Map<TypeReference, String> TYPE_REFERENCE_TO_SIGNATURE =
      Map.ofEntries(
          Map.entry(ARRAY.getDeclaringClass(), ARRAY_SIGNATURE),
          Map.entry(ZEROS.getDeclaringClass(), ZEROS_SIGNATURE),
          Map.entry(ONES.getDeclaringClass(), ONES_SIGNATURE),
          Map.entry(NDARRAY_CONSTRUCTOR.getDeclaringClass(), NDARRAY_CONSTRUCTOR_SIGNATURE),
          Map.entry(EYE.getDeclaringClass(), EYE_SIGNATURE),
          Map.entry(UNIQUE.getDeclaringClass(), UNIQUE_SIGNATURE),
          Map.entry(UNIQUE_VALUES.getDeclaringClass(), UNIQUE_VALUES_SIGNATURE),
          Map.entry(UNIQUE_INDICES.getDeclaringClass(), UNIQUE_INDICES_SIGNATURE),
          Map.entry(RESHAPE.getDeclaringClass(), RESHAPE_SIGNATURE),
          Map.entry(RESHAPE_METHOD.getDeclaringClass(), RESHAPE_METHOD_SIGNATURE),
          Map.entry(ASTYPE.getDeclaringClass(), ASTYPE_SIGNATURE),
          Map.entry(TOLIST.getDeclaringClass(), TOLIST_SIGNATURE));

  public static final FieldReference FLOAT_32 =
      FieldReference.findOrCreate(
          PythonTypes.Root,
          findOrCreateAsciiAtom(DType.FLOAT32.name().toLowerCase(Locale.ROOT)),
          D_TYPE);

  public static final FieldReference FLOAT_64 =
      FieldReference.findOrCreate(
          PythonTypes.Root,
          findOrCreateAsciiAtom(DType.FLOAT64.name().toLowerCase(Locale.ROOT)),
          D_TYPE);

  public static final FieldReference INT_32 =
      FieldReference.findOrCreate(
          PythonTypes.Root,
          findOrCreateAsciiAtom(DType.INT32.name().toLowerCase(Locale.ROOT)),
          D_TYPE);

  public static final FieldReference INT_64 =
      FieldReference.findOrCreate(
          PythonTypes.Root,
          findOrCreateAsciiAtom(DType.INT64.name().toLowerCase(Locale.ROOT)),
          D_TYPE);

  public static final FieldReference UINT_8 =
      FieldReference.findOrCreate(
          PythonTypes.Root,
          findOrCreateAsciiAtom(DType.UINT8.name().toLowerCase(Locale.ROOT)),
          D_TYPE);

  public static final FieldReference STRING =
      FieldReference.findOrCreate(
          PythonTypes.Root,
          findOrCreateAsciiAtom(DType.STRING.name().toLowerCase(Locale.ROOT)),
          D_TYPE);

  /** A mapping from a field reference to its associated {@link DType}, if any. */
  public static final Map<FieldReference, DType> FIELD_REFERENCE_TO_DTYPE =
      Map.ofEntries(
          Map.entry(FLOAT_32, DType.FLOAT32),
          Map.entry(FLOAT_64, DType.FLOAT64),
          Map.entry(INT_32, DType.INT32),
          Map.entry(INT_64, DType.INT64),
          Map.entry(UINT_8, DType.UINT8),
          Map.entry(STRING, DType.STRING));

  private NumpyTypes() {}
}
