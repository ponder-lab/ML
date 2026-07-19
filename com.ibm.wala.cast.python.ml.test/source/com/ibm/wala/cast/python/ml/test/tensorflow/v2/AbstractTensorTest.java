package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.COMPLEX64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorType.mnistInput;
import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static java.util.Collections.emptyMap;
import static java.util.Collections.emptySet;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toSet;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.lsp.AnalysisError;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.cast.python.ml.types.SparseTensorType;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.core.util.io.FileProvider;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.Context;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConcreteTypeKey;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.intset.OrdinalSet;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Shared base for the TensorFlow tensor-type tests (wala/ML#635): the tensor-type constants used by
 * the feature-area test classes in this package. The {@code test(...)} harness overloads and
 * per-test helpers still live in {@code TestTensorflow2Model} and move here as the split proceeds.
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
  protected static final Logger LOGGER = Logger.getLogger(AbstractTensorTest.class.getName());

  /**
   * The largest call graph, in nodes, whose per-node FINE dump is emitted. Above this, only the
   * node count is logged. Large graphs (e.g., nlpgnn) would otherwise emit gigabytes of log output;
   * see <a href="https://github.com/wala/ML/issues/697">wala/ML#697</a>.
   */
  private static final int CALL_GRAPH_DUMP_NODE_LIMIT = 10_000;

  /**
   * Mirrors the recursion in Hybridize's {@code Function.containsPrimitive(InstanceKey, boolean,
   * PointerAnalysis, Set, IProgressMonitor)}: returns {@code true} iff a non-boolean primitive
   * {@link ConstantKey} value is reachable from {@code ik} through transitive instance field PTS
   * lookup. Used by the {@link #testRecursionTensorOnlyHasNoPrimitiveInPTS} regression test to
   * assert the underlying PA state Hybridize consumes.
   *
   * @param ik The {@link InstanceKey} to inspect.
   * @param pa The {@link PointerAnalysis} for instance-field PTS lookups.
   * @param seen The cycle-guard set of already-visited instance keys.
   * @return {@code true} iff a primitive ConstantKey is reachable.
   */
  protected static boolean containsPrimitiveByFieldWalk(
      InstanceKey ik, PointerAnalysis<InstanceKey> pa, Set<InstanceKey> seen) {
    if (!seen.add(ik)) return false;
    if (ik instanceof ConstantKey<?>) {
      Object v = ((ConstantKey<?>) ik).getValue();
      if (v == null) return false;
      // Match Hybridize's "ignore booleans" mode: bool literals don't count as primitive
      // for the purpose of `getHasPrimitiveParameter`.
      if (v.equals(Boolean.TRUE) || v.equals(Boolean.FALSE)) return false;
      return true;
    }
    if (!(ik instanceof AllocationSiteInNode || ik instanceof ConcreteTypeKey)) return false;
    InstanceKey toProcess = ik;
    if (ik instanceof AllocationSiteInNode) {
      // Already a concrete alloc; use as-is.
    }
    for (IField field : toProcess.concreteType().getAllInstanceFields()) {
      InstanceFieldPointerKey fk =
          (InstanceFieldPointerKey)
              pa.getHeapModel().getPointerKeyForInstanceField(toProcess, field);
      OrdinalSet<InstanceKey> fieldPTS = pa.getPointsToSet(fk);
      if (fieldPTS == null) continue;
      for (InstanceKey k : fieldPTS) {
        if (containsPrimitiveByFieldWalk(k, pa, seen)) return true;
      }
    }
    return false;
  }

  protected void test(
      String filename,
      String functionName,
      int expectedNumberOfTensorParameters,
      int expectedNumberOfTensorVariables)
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {filename},
        filename,
        functionName,
        "",
        expectedNumberOfTensorParameters,
        expectedNumberOfTensorVariables,
        emptyMap());
  }

  protected void test(
      String filename,
      String functionName,
      int expectedNumberOfTensorParameters,
      int expectedNumberOfTensorVariables,
      Map<Integer, Set<TensorType>> expectedTensorParameterValueNumberToTypes)
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {filename},
        filename,
        functionName,
        "",
        expectedNumberOfTensorParameters,
        expectedNumberOfTensorVariables,
        expectedTensorParameterValueNumberToTypes);
  }

  /**
   * Single-file test helper with a configurable k-CFA depth. Delegates to the core depth-aware
   * {@code test(...)}; used by tests that need deeper context sensitivity than the default
   * (wala/ML#379, wala/ML#530).
   *
   * @param targetedCfaDepth The k-CFA depth for the targeted context selector.
   * @param filename The file declaring the function under test.
   * @param functionName The function whose tensor types are checked.
   * @param expectedNumberOfTensorParameters The expected number of tensor parameters.
   * @param expectedNumberOfTensorVariables The expected number of function-local tensor variables.
   * @param expectedTensorParameterValueNumberToTypes The expected per-parameter tensor types.
   */
  protected void test(
      int targetedCfaDepth,
      String filename,
      String functionName,
      int expectedNumberOfTensorParameters,
      int expectedNumberOfTensorVariables,
      Map<Integer, Set<TensorType>> expectedTensorParameterValueNumberToTypes)
      throws ClassHierarchyException, CancelException, IOException {
    test(
        targetedCfaDepth,
        new String[] {filename},
        filename,
        functionName,
        "",
        expectedNumberOfTensorParameters,
        expectedNumberOfTensorVariables,
        expectedTensorParameterValueNumberToTypes);
  }

  protected void test(
      String[] projectFilenames,
      String filename,
      String functionName,
      String pythonPath,
      int expectedNumberOfTensorParameters,
      int expectedNumberOfFunctionTensorVariables,
      Map<Integer, Set<TensorType>> expectedTensorParameterValueNumberToTypes)
      throws ClassHierarchyException, CancelException, IOException {
    test(
        PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH,
        projectFilenames,
        filename,
        functionName,
        pythonPath,
        expectedNumberOfTensorParameters,
        expectedNumberOfFunctionTensorVariables,
        expectedTensorParameterValueNumberToTypes);
  }

  /**
   * Core test helper with a configurable k-CFA depth. Most tests use the default-depth overloads;
   * tests that exercise per-context layer-output precision (e.g. {@code testNeuralNetwork*}) opt
   * into a deeper depth so a user model invoked from multiple sites does not collapse its
   * layer-output allocations (wala/ML#379, wala/ML#530).
   *
   * <p>Both checked quantities are <em>source-level</em>, deduplicated across calling contexts
   * (wala/ML#371, Option 2): the tensor-variable count is the number of distinct value numbers in
   * the function under test, and the per-parameter type assertion is the union of each value
   * number's types across all contexts. Under k-CFA the same IR variable appears once per calling
   * context, so neither quantity is coupled to the context-sensitivity depth &mdash; context
   * multiplicity is analysis bookkeeping, not a source-level property, and matches the downstream
   * consumer ({@code @tf.function(input_signature=...)}), which is indexed by source-level
   * parameter position.
   *
   * @param targetedCfaDepth The k-CFA depth for the targeted context selector.
   * @param projectFilenames The script module file names making up the project.
   * @param filename The file declaring the function under test.
   * @param functionName The function whose tensor types are checked.
   * @param pythonPath The Python path root for module resolution.
   * @param expectedNumberOfTensorParameters The expected number of tensor parameters.
   * @param expectedNumberOfFunctionTensorVariables The expected number of function-local tensor
   *     variables, counted source-level (distinct value numbers, deduplicated across contexts).
   * @param expectedTensorParameterValueNumberToTypes The expected per-parameter tensor types.
   */
  protected void test(
      int targetedCfaDepth,
      String[] projectFilenames,
      String filename,
      String functionName,
      String pythonPath,
      int expectedNumberOfTensorParameters,
      int expectedNumberOfFunctionTensorVariables,
      Map<Integer, Set<TensorType>> expectedTensorParameterValueNumberToTypes)
      throws ClassHierarchyException, CancelException, IOException {
    List<File> pathFiles = this.getPathFiles(pythonPath);
    PythonTensorAnalysisEngine engine = makeEngine(targetedCfaDepth, pathFiles, projectFilenames);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();

    addPytestEntrypoints(builder);

    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);

    if (LOGGER.isLoggable(Level.FINE)) {
      // Both the IR dump (`dumpCG`) and the per-node call-graph dump render each node's context,
      // whose scope-mapping/receiver contexts recurse and materialize gigantic strings on large
      // graphs (e.g., nlpgnn). Gate the whole dump behind a node-count limit rather than via
      // CallGraph.toString(), whose monolithic materialization exhausts the heap; see
      // https://github.com/wala/ML/issues/697.
      int nodeCount = CG.getNumberOfNodes();
      if (nodeCount <= CALL_GRAPH_DUMP_NODE_LIMIT) {
        CAstCallGraphUtil.AVOID_DUMP.set(false);
        CAstCallGraphUtil.dumpCG(
            ((SSAPropagationCallGraphBuilder) builder).getCFAContextInterpreter(),
            builder.getPointerAnalysis(),
            CG);
        LOGGER.fine("Call graph has " + nodeCount + " node(s):");
        // Render each node by number and method signature rather than CGNode.toString(), which
        // renders the node's context and can trigger the same runaway recursion; see
        // https://github.com/wala/ML/issues/697.
        for (CGNode node : CG)
          LOGGER.fine(() -> CG.getNumber(node) + ": " + node.getMethod().getSignature());
      } else
        LOGGER.fine(
            "Call graph has "
                + nodeCount
                + " node(s); dump skipped (limit "
                + CALL_GRAPH_DUMP_NODE_LIMIT
                + ").");
    }

    TensorTypeAnalysis analysis = engine.performAnalysis(builder);
    // A lazy FINE supplier: the whole-lattice dump is a multi-hundred-MB string on the
    // whole-project fixtures, and an eager INFO concatenation builds it even when the CI logging
    // config discards the message (the release runner's heap exhaustion).
    LOGGER.fine(() -> "Tensor analysis: " + analysis);

    Map<PointerKey, AnalysisError> errors = engine.getErrors();

    errors.forEach(
        (k, v) ->
            LOGGER.info(
                () -> "Pointer key: " + k + " has analysis error: " + v + " at " + v.position()));

    // a mapping from function signatures to pointer keys.
    Map<String, Set<LocalPointerKey>> functionSignatureToPointerKeys = new HashMap<>();

    // a mapping from function signatures to tensor variables.
    Map<String, Set<TensorVariable>> functionSignatureToTensorVariables = new HashMap<>();

    // a mapping from pointer keys to tensor variables.
    Map<PointerKey, TensorVariable> pointerKeyToTensorVariable = new HashMap<>();

    // for each pointer key, tensor variable pair.
    analysis.forEach(
        pt -> {
          PointerKey pointerKey = pt.fst;
          TensorVariable tensorVariable = pt.snd;

          // associate the pointer key to the tensor variable.
          pointerKeyToTensorVariable.put(pointerKey, tensorVariable);

          if (pointerKey instanceof LocalPointerKey) {
            LocalPointerKey localPointerKey = (LocalPointerKey) pointerKey;

            // get the call graph node associated with the pointer key.
            CGNode node = localPointerKey.getNode();

            // get the method associated with the call graph node.
            IMethod method = node.getMethod();
            String methodSignature = method.getSignature();

            // associate the method to the pointer key.
            functionSignatureToPointerKeys.compute(
                methodSignature,
                (_, v) -> {
                  if (v == null) v = new HashSet<>();
                  v.add(localPointerKey);
                  return v;
                });

            // associate the method to the tensor variables.
            functionSignatureToTensorVariables.compute(
                methodSignature,
                (_, v) -> {
                  if (v == null) v = new HashSet<>();
                  v.add(tensorVariable);
                  return v;
                });
          } else LOGGER.warning(() -> "Encountered: " + pointerKey.getClass());
        });

    final String functionSignature =
        "script " + filename.replace('/', '.') + "." + functionName + ".do()LRoot;";

    // List the CG nodes as a "flat" list.
    LOGGER.fine(
        () ->
            "Call graph nodes:\n"
                + getFunctionSignatures(CG).collect(Collectors.joining("\n\t", "\t", "")));

    // check that the function exists in the call graph.
    assertTrue(
        "Function must exist in call graph.",
        getFunctionSignatures(CG).anyMatch(s -> s.equals(functionSignature)));

    // get the tensor variables for the function.
    LOGGER.fine("Looking for signature: " + functionSignature);
    LOGGER.fine(
        "Available signatures in functionSignatureToTensorVariables: "
            + functionSignatureToTensorVariables.keySet());

    // Dump `(vn, TensorVariable)` pairs for the FUT so cross-branch comparison can identify
    // which specific SSA value explains a count discrepancy.
    LOGGER.fine(
        () ->
            "Tensor variables for "
                + functionSignature
                + ":\n\t"
                + functionSignatureToPointerKeys
                    .getOrDefault(functionSignature, emptySet())
                    .stream()
                    .sorted(Comparator.comparingInt(LocalPointerKey::getValueNumber))
                    .map(
                        pk ->
                            "vn="
                                + pk.getValueNumber()
                                + " -> "
                                + pointerKeyToTensorVariable.get(pk)
                                + " [ctx: "
                                + summarizeContext(pk.getNode())
                                + "]")
                    .collect(Collectors.joining("\n\t")));

    // Count tensor variables at the source level: one per distinct value number, deduplicated
    // across calling contexts (wala/ML#371). Under k-CFA the same IR variable (same `vn` in the
    // same method) appears once per calling context,
    // so counting `(CGNode, vn)` pairs would couple the expected count to the context-sensitivity
    // depth. The downstream consumer (`@tf.function(input_signature=...)`) is indexed by
    // source-level parameter position, and the helper's type assertion already unions per `vn`
    // across contexts; counting distinct `vn`s keeps the count axis consistent with that framing
    // and stable as the depth changes (context multiplicity is analysis bookkeeping, not a
    // source-level property).
    Set<Integer> functionTensorValueNumbers =
        functionSignatureToPointerKeys.getOrDefault(functionSignature, emptySet()).stream()
            .map(LocalPointerKey::getValueNumber)
            .collect(toSet());

    assertEquals(expectedNumberOfFunctionTensorVariables, functionTensorValueNumbers.size());

    // check value number cardinality.
    assertEquals(
        "Each tensor parameter should have a unique value number.",
        expectedNumberOfTensorParameters,
        expectedTensorParameterValueNumberToTypes.size());

    // get the pointer keys for the function by their contexts.
    Map<Context, Set<LocalPointerKey>> contextToFunctionParameterPointerKeys =
        functionSignatureToPointerKeys.getOrDefault(functionSignature, emptySet()).stream()
            .filter(LocalPointerKey::isParameter)
            .collect(groupingBy(lpk -> lpk.getNode().getContext(), toSet()));

    assertTrue(
        "Because tensor parameters are inferred via function arguments, we need at least one"
            + " calling context if we are expecting at least one tensor parameter.",
        expectedNumberOfTensorParameters <= 0 || contextToFunctionParameterPointerKeys.size() > 0);

    // Union actual types per source-level vn across all contexts. The downstream consumer for this
    // analysis is `@tf.function(input_signature=...)`, which is indexed by source-level parameter
    // position — so the comparison that matches the use case is "union of per-context actuals for
    // vn equals expected set", not "every context individually contains the full expected set".
    // The same framing grounds wala/ML#371 Option 2 on the count axis.
    Map<Integer, Set<TensorType>> actualTypesByValueNumber = new HashMap<>();

    for (Context ctx : contextToFunctionParameterPointerKeys.keySet()) {
      Set<LocalPointerKey> functionParameterPointerKeys =
          contextToFunctionParameterPointerKeys.get(ctx);

      // accumulate per-vn types across contexts.
      for (LocalPointerKey lpk : functionParameterPointerKeys) {
        TensorVariable tensorVariable = pointerKeyToTensorVariable.get(lpk);
        assertNotNull("Checking tensor variable for pointer key: " + lpk + ".", tensorVariable);

        Set<TensorType> types = tensorVariable.getTypes();
        assertNotNull("Checking tensor variable for pointer key: " + lpk + ".", types);

        actualTypesByValueNumber
            .computeIfAbsent(lpk.getValueNumber(), k -> new HashSet<>())
            .addAll(types);
      }
    }

    // Check the tensor-parameter count and value numbers on the union across contexts, mirroring
    // the function-variable count above: with receiver-keyed contexts (wala/ML#679), a single
    // source-level call chain fans out into finer contexts and not every context sees every tensor
    // argument, but the downstream consumer is indexed by source-level parameter position, so the
    // source-level property is the per-vn union. A parameter that loses typing in *every* context
    // still fails here.
    assertEquals(expectedNumberOfTensorParameters, actualTypesByValueNumber.size());
    assertEquals(
        expectedTensorParameterValueNumberToTypes.keySet(), actualTypesByValueNumber.keySet());

    // compare expected against the union across contexts, per vn.
    for (Map.Entry<Integer, Set<TensorType>> entry :
        expectedTensorParameterValueNumberToTypes.entrySet()) {
      int vn = entry.getKey();
      Set<TensorType> expectedTypes = entry.getValue();
      Set<TensorType> actualUnion = actualTypesByValueNumber.getOrDefault(vn, emptySet());

      if (LOGGER.isLoggable(Level.INFO) && !expectedTypes.equals(actualUnion)) {
        LOGGER.info("Type-union mismatch for value number: " + vn + ".");
        LOGGER.info("  Expected: " + expectedTypes);
        LOGGER.info("  Actual (union across contexts): " + actualUnion);
      }

      assertEquals(
          "Comparing expected types for value number: " + vn + " (union across contexts).",
          expectedTypes,
          actualUnion);
    }
  }

  /** Matches the script-relative method identifiers a Python {@link Context} chain mentions. */
  private static final Pattern CONTEXT_METHOD_PATTERN =
      Pattern.compile("(?:tests|nlpgnn|src|proj[0-9]*)/[A-Za-z0-9_/.-]+\\.py/[A-Za-z0-9_]+");

  /**
   * Summarizes a node's calling context as the distinct script-relative method identifiers its
   * context chain mentions, so the per-context tensor-variable dump identifies which caller
   * pipeline contributed each value without printing the full recursive context.
   *
   * @param node The {@link CGNode} whose context is summarized.
   * @return A comma-separated summary of the context's method identifiers.
   */
  private static String summarizeContext(CGNode node) {
    return CONTEXT_METHOD_PATTERN
        .matcher(node.getContext().toString())
        .results()
        .map(MatchResult::group)
        .distinct()
        .collect(Collectors.joining(", "));
  }

  /**
   * Returns a {@link Stream} of {@link String}s representing the signatures of functions
   * represented by the nodes in the given {@link CallGraph}.
   *
   * @param CG The {@link CallGraph} containing the nodes in question.
   * @return A {@link Stream} of {@link String}s representing the signatures of functions
   *     represented by the nodes in the given {@link CallGraph}.
   */
  private static Stream<String> getFunctionSignatures(CallGraph CG) {
    return CG.stream().map(CGNode::getMethod).map(IMethod::getSignature);
  }

  /**
   * Extracts a {@link List} of {@link File}s from the given {@link String} representing a list of
   * paths. Each path is separated by a colon.
   *
   * @param string A colon-separated list of paths.
   * @return {@link List} of {@link File}s constructed by parsing the given {@link String}.
   */
  protected List<File> getPathFiles(String string) {
    if (string == null || string.isEmpty() || string.isBlank()) return emptyList();

    return Arrays.asList(string.split(":")).stream()
        .map(
            s -> {
              File f = new File(s);

              if (f.exists()) return f;

              try {
                URL url = new URI(s).toURL();
                return new File(new FileProvider().filePathFromURL(url));
              } catch (MalformedURLException | URISyntaxException | IllegalArgumentException e) {
                try {
                  URL resource = this.getClass().getResource("/" + string);
                  String path = resource.getPath();
                  return new File(path);
                } catch (Exception e1) {
                  throw new RuntimeException(e1);
                }
              }
            })
        .collect(toList());
  }
}
