package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.COMPLEX64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorType.mnistInput;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;

import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.cast.python.ml.types.SparseTensorType;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;

/**
 * Shared base for the TensorFlow tensor-type tests (wala/ML#635): the tensor-type constants used by
 * the feature-area test classes in this package. The {@code test(...)} harness overloads and
 * per-test helpers still live in {@link TestTensorflow2Model} and move here as the split proceeds.
 */
public abstract class AbstractTensorTest extends TestPythonMLCallGraphShape {

  protected static final String FLOAT_32 = FLOAT32.name().toLowerCase();

  protected static final String COMPLEX_64 = COMPLEX64.name().toLowerCase();

  protected static final String COMPLEX_128 = DType.COMPLEX128.name().toLowerCase();

  protected static final String FLOAT_64 = FLOAT64.name().toLowerCase();

  protected static final String INT_32 = INT32.name().toLowerCase();

  protected static final String INT_64 = DType.INT64.name().toLowerCase();

  protected static final String UINT_8 = DType.UINT8.name().toLowerCase();

  protected static final String BOOL = DType.BOOL.name().toLowerCase();

  protected static final String STRING = DType.STRING.name().toLowerCase();

  protected static final String OBJECT = DType.OBJECT.name().toLowerCase();

  protected static final String UNKNOWN = DType.UNKNOWN.name().toLowerCase();

  protected static final TensorType MNIST_INPUT = mnistInput();

  protected static final TensorType SCALAR_TENSOR_OF_INT32 = new TensorType(INT_32, emptyList());

  protected static final TensorType SCALAR_TENSOR_OF_INT64 = new TensorType(INT_64, emptyList());

  protected static final TensorType SCALAR_TENSOR_OF_FLOAT32 =
      new TensorType(FLOAT_32, emptyList());

  protected static final TensorType SCALAR_TENSOR_OF_STRING = new TensorType(STRING, emptyList());

  protected static final TensorType SCALAR_TENSOR_OF_BOOL = new TensorType(BOOL, emptyList());

  protected static final TensorType TENSOR_3_BOOL = TensorType.of(BOOL, 3);

  protected static final TensorType TENSOR_1_1_FLOAT32 = TensorType.of(FLOAT_32, 1, 1);

  protected static final TensorType TENSOR_2_3_3_FLOAT32 = TensorType.of(FLOAT_32, 2, 3, 3);

  protected static final TensorType TENSOR_0_0_FLOAT32 = TensorType.of(FLOAT_32, 0, 0);

  protected static final TensorType TENSOR_1_2_FLOAT32 = TensorType.of(FLOAT_32, 1, 2);

  protected static final TensorType TENSOR_1_5_FLOAT32 = TensorType.of(FLOAT_32, 1, 5);

  protected static final TensorType TENSOR_1_10_FLOAT32 = TensorType.of(FLOAT_32, 1, 10);

  protected static final TensorType TENSOR_1_3_FLOAT32 = TensorType.of(FLOAT_32, 1, 3);

  protected static final TensorType TENSOR_3_1_FLOAT32 = TensorType.of(FLOAT_32, 3, 1);

  @SuppressWarnings("unused")
  protected static final TensorType TENSOR_32_INT32 = TensorType.of(INT_32, 32);

  protected static final TensorType TENSOR_32_UINT8 = TensorType.of(UINT_8, 32);

  protected static final TensorType TENSOR_16_UINT8 = TensorType.of(UINT_8, 16);

  protected static final TensorType TENSOR_256_784_FLOAT32 = TensorType.of(FLOAT_32, 256, 784);

  protected static final TensorType TENSOR_256_28_28_FLOAT32 = TensorType.of(FLOAT_32, 256, 28, 28);

  protected static final TensorType TENSOR_10000_784_FLOAT32 = TensorType.of(FLOAT_32, 10000, 784);

  protected static final TensorType TENSOR_5_784_FLOAT32 = TensorType.of(FLOAT_32, 5, 784);

  protected static final TensorType TENSOR_60000_784_UINT8 = TensorType.of(UINT_8, 60000, 784);

  protected static final TensorType TENSOR_256_10_FLOAT32 = TensorType.of(FLOAT_32, 256, 10);

  protected static final TensorType TENSOR_256_UINT8 = TensorType.of(UINT_8, 256);

  protected static final TensorType TENSOR_10000_10_FLOAT32 = TensorType.of(FLOAT_32, 10000, 10);

  protected static final TensorType TENSOR_10000_UINT8 = TensorType.of(UINT_8, 10000);

  protected static final TensorType TENSOR_32_28_28_UINT8 = TensorType.of(UINT_8, 32, 28, 28);

  protected static final TensorType TENSOR_5_28_28_UINT8 = TensorType.of(UINT_8, 5, 28, 28);

  protected static final TensorType TENSOR_3_28_28_UINT8 = TensorType.of(UINT_8, 3, 28, 28);

  protected static final TensorType TENSOR_1_2_INT32 = TensorType.of(INT_32, 1, 2);

  protected static final TensorType TENSOR_1_5_INT32 = TensorType.of(INT_32, 1, 5);

  protected static final TensorType TENSOR_1_10_INT32 = TensorType.of(INT_32, 1, 10);

  protected static final TensorType TENSOR_2_2_FLOAT32 = TensorType.of(FLOAT_32, 2, 2);

  protected static final TensorType TENSOR_NONE_32_FLOAT32 =
      new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(32)));

  protected static final TensorType TENSOR_NONE_3_FLOAT32 =
      new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(3)));

  protected static final TensorType TENSOR_NONE_4_FLOAT32 =
      new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(4)));

  protected static final TensorType TENSOR_NONE_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(2)));

  protected static final TensorType TENSOR_NONE_NONE_STRING =
      new TensorType(STRING, asList(DynamicDim.INSTANCE, DynamicDim.INSTANCE));

  protected static final TensorType TENSOR_4_RAGGED_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_3_RAGGED_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_4_RAGGED_RAGGED_NONE_STRING =
      new TensorType(
          STRING,
          asList(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, DynamicDim.INSTANCE));

  protected static final TensorType TENSOR_3_RAGGED_RAGGED_STRING =
      new TensorType(STRING, asList(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_1_RAGGED_RAGGED_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(1), RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_2_RAGGED_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_2_2_INT32 = TensorType.of(INT_32, 2, 2);

  protected static final TensorType TENSOR_3_2_FLOAT32 = TensorType.of(FLOAT_32, 3, 2);

  protected static final TensorType TENSOR_2_4_3_FLOAT32 = TensorType.of(FLOAT_32, 2, 4, 3);

  protected static final TensorType TENSOR_4_3_2_FLOAT32 = TensorType.of(FLOAT_32, 4, 3, 2);

  protected static final TensorType TENSOR_4_3_FLOAT32 = TensorType.of(FLOAT_32, 4, 3);

  protected static final TensorType TENSOR_2_3_FLOAT32 = TensorType.of(FLOAT_32, 2, 3);

  protected static final TensorType TENSOR_1_1_3_2_FLOAT32 = TensorType.of(FLOAT_32, 1, 1, 3, 2);

  protected static final TensorType TENSOR_2_3_1_FLOAT32 = TensorType.of(FLOAT_32, 2, 3, 1);

  protected static final TensorType TENSOR_2_3_FLOAT64 = TensorType.of(FLOAT_64, 2, 3);

  protected static final TensorType TENSOR_4_INT64 = TensorType.of(INT_64, 4);

  protected static final TensorType TENSOR_100_784_FLOAT32 = TensorType.of(FLOAT_32, 100, 784);

  protected static final TensorType TENSOR_4_8_FLOAT32 = TensorType.of(FLOAT_32, 4, 8);

  protected static final TensorType TENSOR_4_512_FLOAT32 = TensorType.of(FLOAT_32, 4, 512);

  protected static final TensorType TENSOR_2_64_FLOAT32 = TensorType.of(FLOAT_32, 2, 64);

  protected static final TensorType SPARSE_TENSOR_4_4_FLOAT32 =
      new SparseTensorType(FLOAT32, asList(new NumericDim(4), new NumericDim(4)));

  protected static final TensorType TENSOR_4_10_FLOAT32 = TensorType.of(FLOAT_32, 4, 10);

  protected static final TensorType TENSOR_4_1_INT32 = TensorType.of(INT_32, 4, 1);

  protected static final TensorType TENSOR_256_256_3_FLOAT32 = TensorType.of(FLOAT_32, 256, 256, 3);

  protected static final TensorType TENSOR_2_3_4_FLOAT32 = TensorType.of(FLOAT_32, 2, 3, 4);

  protected static final TensorType TENSOR_2_2_4_FLOAT32 = TensorType.of(FLOAT_32, 2, 2, 4);

  protected static final TensorType TENSOR_2_2_1_FLOAT32 = TensorType.of(FLOAT_32, 2, 2, 1);

  protected static final TensorType TENSOR_2_5_6_FLOAT32 = TensorType.of(FLOAT_32, 2, 5, 6);

  protected static final TensorType TENSOR_4_4_6_FLOAT32 = TensorType.of(FLOAT_32, 4, 4, 6);

  protected static final TensorType TENSOR_4_6_FLOAT32 = TensorType.of(FLOAT_32, 4, 6);

  protected static final TensorType TENSOR_1_2_2_27_FLOAT32 = TensorType.of(FLOAT_32, 1, 2, 2, 27);

  protected static final TensorType TENSOR_4_4_FLOAT32 = TensorType.of(FLOAT_32, 4, 4);

  protected static final TensorType TENSOR_2_5_INT32 = TensorType.of(INT_32, 2, 5);

  protected static final TensorType TENSOR_3_3_FLOAT32 = TensorType.of(FLOAT_32, 3, 3);

  protected static final TensorType TENSOR_3_3_INT32 = TensorType.of(INT_32, 3, 3);

  protected static final TensorType TENSOR_0_RAGGED_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(0), RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_0_RAGGED_3_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(0), RaggedDim.INSTANCE, new NumericDim(3)));

  protected static final TensorType TENSOR_1_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(1), RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_1_RAGGED_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(1), RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_2_NONE_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), DynamicDim.INSTANCE));

  protected static final TensorType TENSOR_2_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_NONE_RAGGED_INT32 =
      new TensorType(INT_32, asList(DynamicDim.INSTANCE, RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_2_RAGGED_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(2), RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_2_RAGGED_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(2), RaggedDim.INSTANCE, new NumericDim(2)));

  protected static final TensorType TENSOR_2_RAGGED_2_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), RaggedDim.INSTANCE, new NumericDim(2)));

  protected static final TensorType TENSOR_2_RAGGED_2_3_INT32 =
      new TensorType(
          INT_32,
          asList(new NumericDim(2), RaggedDim.INSTANCE, new NumericDim(2), new NumericDim(3)));

  protected static final TensorType TENSOR_2_RAGGED_2_2_INT32 =
      new TensorType(
          INT_32,
          asList(new NumericDim(2), RaggedDim.INSTANCE, new NumericDim(2), new NumericDim(2)));

  protected static final TensorType TENSOR_2_RAGGED_RAGGED_RAGGED_FLOAT32 =
      new TensorType(
          FLOAT_32,
          asList(new NumericDim(2), RaggedDim.INSTANCE, RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_3_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(3), RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_3_RAGGED_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_4_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(4), RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_3_RAGGED_RAGGED_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_3_RAGGED_1_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), RaggedDim.INSTANCE, new NumericDim(1)));

  protected static final TensorType TENSOR_2_3_INT32 = TensorType.of(INT_32, 2, 3);

  protected static final TensorType TENSOR_2_4_FLOAT32 = TensorType.of(FLOAT_32, 2, 4);

  protected static final TensorType TENSOR_2_6_INT32 = TensorType.of(INT_32, 2, 6);

  protected static final TensorType TENSOR_2_1_FLOAT32 = TensorType.of(FLOAT_32, 2, 1);

  protected static final TensorType TENSOR_10_2_FLOAT32 = TensorType.of(FLOAT_32, 10, 2);

  protected static final TensorType TENSOR_10_2_FLOAT64 = TensorType.of(FLOAT_64, 10, 2);

  protected static final TensorType TENSOR_5_2_FLOAT32 = TensorType.of(FLOAT_32, 5, 2);

  protected static final TensorType TENSOR_5_2_INT32 = TensorType.of(INT_32, 5, 2);

  protected static final TensorType TENSOR_5_5_FLOAT32 = TensorType.of(FLOAT_32, 5, 5);

  protected static final TensorType TENSOR_5_5_INT32 = TensorType.of(INT_32, 5, 5);

  protected static final TensorType TENSOR_5_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(5), RaggedDim.INSTANCE));

  protected static final TensorType TENSOR_2_3_3_INT32 = TensorType.of(INT_32, 2, 3, 3);

  protected static final TensorType TENSOR_2_3_4_INT32 = TensorType.of(INT_32, 2, 3, 4);

  protected static final TensorType TENSOR_2_5_3_FLOAT32 = TensorType.of(FLOAT_32, 2, 5, 3);

  protected static final TensorType TENSOR_3_2_2_FLOAT32 = TensorType.of(FLOAT_32, 3, 2, 2);

  protected static final TensorType TENSOR_5_6_FLOAT32 = TensorType.of(FLOAT_32, 5, 6);

  protected static final TensorType TENSOR_30_FLOAT32 = TensorType.of(FLOAT_32, 30);

  protected static final TensorType TENSOR_4_5_6_FLOAT32 = TensorType.of(FLOAT_32, 4, 5, 6);

  protected static final TensorType TENSOR_7_5_2_FLOAT32 = TensorType.of(FLOAT_32, 7, 5, 2);

  protected static final TensorType TENSOR_30_3_2_FLOAT32 = TensorType.of(FLOAT_32, 30, 3, 2);

  protected static final TensorType TENSOR_3_2_2_3_FLOAT32 = TensorType.of(FLOAT_32, 3, 2, 2, 3);

  protected static final TensorType TENSOR_2_2_2_3_FLOAT32 = TensorType.of(FLOAT_32, 2, 2, 2, 3);

  protected static final TensorType TENSOR_20_28_28_FLOAT32 = TensorType.of(FLOAT_32, 20, 28, 28);

  protected static final TensorType TENSOR_20_28_28_INT32 = TensorType.of(INT_32, 20, 28, 28);

  protected static final TensorType TENSOR_20_10_FLOAT32 = TensorType.of(FLOAT_32, 20, 10);

  protected static final TensorType TENSOR_20_64_FLOAT32 = TensorType.of(FLOAT_32, 20, 64);

  protected static final TensorType TENSOR_60000_28_28_FLOAT32 =
      TensorType.of(FLOAT_32, 60000, 28, 28);

  protected static final TensorType TENSOR_60000_28_28_UINT8 = TensorType.of(UINT_8, 60000, 28, 28);

  protected static final TensorType TENSOR_50000_32_32_3_UINT8 =
      TensorType.of(UINT_8, 50000, 32, 32, 3);

  protected static final TensorType TENSOR_8982_INT64 = TensorType.of(INT_64, 8982);

  protected static final TensorType TENSOR_404_13_FLOAT64 = TensorType.of(FLOAT_64, 404, 13);

  protected static final TensorType TENSOR_404_FLOAT64 = TensorType.of(FLOAT_64, 404);

  protected static final TensorType TENSOR_60000_UINT8 = TensorType.of(UINT_8, 60000);

  protected static final TensorType TENSOR_50000_1_UINT8 = TensorType.of(UINT_8, 50000, 1);

  protected static final TensorType TENSOR_50000_1_INT64 = TensorType.of(INT_64, 50000, 1);

  protected static final TensorType TENSOR_8982_OBJECT = TensorType.of(OBJECT, 8982);

  protected static final TensorType TENSOR_102_13_FLOAT64 = TensorType.of(FLOAT_64, 102, 13);

  protected static final TensorType TENSOR_102_FLOAT64 = TensorType.of(FLOAT_64, 102);

  protected static final TensorType TENSOR_10000_32_32_3_UINT8 =
      TensorType.of(UINT_8, 10000, 32, 32, 3);

  protected static final TensorType TENSOR_10000_1_UINT8 = TensorType.of(UINT_8, 10000, 1);

  protected static final TensorType TENSOR_10000_1_INT64 = TensorType.of(INT_64, 10000, 1);

  protected static final TensorType TENSOR_10000_28_28_UINT8 = TensorType.of(UINT_8, 10000, 28, 28);

  protected static final TensorType TENSOR_2246_INT64 = TensorType.of(INT_64, 2246);

  protected static final TensorType TENSOR_2246_OBJECT = TensorType.of(OBJECT, 2246);

  /** A {@code float32} tensor whose shape cannot be statically inferred. */
  protected static final TensorType TENSOR_UNKNOWN_SHAPE_FLOAT32 = new TensorType(FLOAT_32, null);

  /** Fully-⊤ tensor type: unknown shape and unknown dtype. */
  protected static final TensorType TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE =
      new TensorType(UNKNOWN, null);

  protected static final TensorType TENSOR_1_FLOAT32 = TensorType.of(FLOAT_32, 1);

  protected static final TensorType TENSOR_2_FLOAT32 = TensorType.of(FLOAT_32, 2);

  protected static final TensorType TENSOR_2_FLOAT64 = TensorType.of(FLOAT_64, 2);

  protected static final TensorType TENSOR_UNRESOLVED_UNRESOLVED_FLOAT32 =
      new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE));

  protected static final TensorType TENSOR_2_INT32 = TensorType.of(INT_32, 2);

  protected static final TensorType TENSOR_2_INT64 = TensorType.of(INT_64, 2);

  protected static final TensorType TENSOR_INT64_UNKNOWN_SHAPE = new TensorType(INT_64, null);

  protected static final TensorType TENSOR_DYNAMIC_INT64 =
      new TensorType(INT_64, asList(DynamicDim.INSTANCE));

  protected static final TensorType TENSOR_INT32_UNKNOWN_SHAPE = new TensorType(INT_32, null);

  protected static final TensorType TENSOR_1_0_0_9_INT32 = TensorType.of(INT_32, 1, 0, 0, 9);

  protected static final TensorType TENSOR_UNKNOWN_SHAPE_BOOL = new TensorType(BOOL, null);

  protected static final TensorType TENSOR_3_INT32 = TensorType.of(INT_32, 3);

  protected static final TensorType TENSOR_3_INT64 = TensorType.of(INT_64, 3);

  protected static final TensorType TENSOR_3_FLOAT32 = TensorType.of(FLOAT_32, 3);

  protected static final TensorType TENSOR_4_FLOAT32 = TensorType.of(FLOAT_32, 4);

  protected static final TensorType TENSOR_2_2_BOOL = TensorType.of(BOOL, 2, 2);

  protected static final TensorType TENSOR_3_5_BOOL = TensorType.of(BOOL, 3, 5);

  protected static final TensorType TENSOR_3_5_INT32 = TensorType.of(INT_32, 3, 5);

  protected static final TensorType TENSOR_3_5_FLOAT32 = TensorType.of(FLOAT_32, 3, 5);

  protected static final TensorType TENSOR_4_FLOAT64 = TensorType.of(FLOAT_64, 4);

  protected static final TensorType TENSOR_5_FLOAT32 = TensorType.of(FLOAT_32, 5);

  protected static final TensorType TENSOR_5_FLOAT64 = TensorType.of(FLOAT_64, 5);

  protected static final TensorType TENSOR_64_5_FLOAT32 = TensorType.of(FLOAT_32, 64, 5);

  protected static final TensorType TENSOR_7_FLOAT32 = TensorType.of(FLOAT_32, 7);

  protected static final TensorType TENSOR_32_7_FLOAT32 = TensorType.of(FLOAT_32, 32, 7);

  protected static final TensorType TENSOR_64_7_FLOAT32 = TensorType.of(FLOAT_32, 64, 7);

  protected static final TensorType TENSOR_20_5_FLOAT32 = TensorType.of(FLOAT_32, 20, 5);

  protected static final TensorType TENSOR_20_7_FLOAT32 = TensorType.of(FLOAT_32, 20, 7);

  protected static final TensorType TENSOR_5_INT32 = TensorType.of(INT_32, 5);

  protected static final TensorType TENSOR_5_INT64 = TensorType.of(INT_64, 5);

  protected static final TensorType TENSOR_4_INT32 = TensorType.of(INT_32, 4);

  protected static final TensorType TENSOR_1_INT32 = TensorType.of(INT_32, 1);

  protected static final TensorType TENSOR_3_4_INT32 = TensorType.of(INT_32, 3, 4);

  protected static final TensorType TENSOR_3_4_FLOAT32 = TensorType.of(FLOAT_32, 3, 4);

  protected static final TensorType TENSOR_4_5_FLOAT32 = TensorType.of(FLOAT_32, 4, 5);

  protected static final TensorType TENSOR_1_28_28_1_FLOAT32 =
      TensorType.of(FLOAT_32, 1, 28, 28, 1);

  protected static final TensorType TENSOR_6_INT32 = TensorType.of(INT_32, 6);

  protected static final TensorType TENSOR_6_FLOAT32 = TensorType.of(FLOAT_32, 6);

  protected static final TensorType TENSOR_256_28_28_1_FLOAT32 =
      TensorType.of(FLOAT_32, 256, 28, 28, 1);

  protected static final TensorType TENSOR_32_28_28_1_FLOAT32 =
      TensorType.of(FLOAT_32, 32, 28, 28, 1);

  protected static final TensorType TENSOR_16_28_28_1_FLOAT32 =
      TensorType.of(FLOAT_32, 16, 28, 28, 1);

  protected static final TensorType TENSOR_256_64_FLOAT32 = TensorType.of(FLOAT_32, 256, 64);

  protected static final TensorType TENSOR_96_28_28_1_FLOAT32 =
      TensorType.of(FLOAT_32, 96, 28, 28, 1);

  protected static final TensorType TENSOR_4096_32_32_3_FLOAT32 =
      TensorType.of(FLOAT_32, 4096, 32, 32, 3);

  protected static final TensorType TENSOR_4096_UINT8 = TensorType.of(UINT_8, 4096);

  protected static final TensorType TENSOR_3_STRING = TensorType.of(STRING, 3);

  protected static final TensorType TENSOR_25000_INT64 = TensorType.of(INT_64, 25000);

  protected static final TensorType TENSOR_25000_OBJECT = TensorType.of(OBJECT, 25000);
}
