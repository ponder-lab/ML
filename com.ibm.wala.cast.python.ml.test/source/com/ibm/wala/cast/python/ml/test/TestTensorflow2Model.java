package com.ibm.wala.cast.python.ml.test;

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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.lsp.AnalysisError;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.BroadcastTo;
import com.ibm.wala.cast.python.ml.client.Constant;
import com.ibm.wala.cast.python.ml.client.Linspace;
import com.ibm.wala.cast.python.ml.client.NpArray;
import com.ibm.wala.cast.python.ml.client.NpOnes;
import com.ibm.wala.cast.python.ml.client.NpZeros;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.SliceBuiltinOperation;
import com.ibm.wala.cast.python.ml.types.SparseTensorType;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
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
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.Test;

/** Test TF2 APIs. */
public class TestTensorflow2Model extends TestPythonMLCallGraphShape {

  private static final Logger LOGGER = Logger.getLogger(TestTensorflow2Model.class.getName());

  /**
   * The largest call graph, in nodes, whose per-node FINE dump is emitted. Above this, only the
   * node count is logged. Large graphs (e.g., nlpgnn) would otherwise emit gigabytes of log output;
   * see <a href="https://github.com/wala/ML/issues/697">wala/ML#697</a>.
   */
  private static final int CALL_GRAPH_DUMP_NODE_LIMIT = 10_000;

  private static final String FLOAT_32 = FLOAT32.name().toLowerCase();

  private static final String COMPLEX_64 = COMPLEX64.name().toLowerCase();

  private static final String COMPLEX_128 = DType.COMPLEX128.name().toLowerCase();

  private static final String FLOAT_64 = FLOAT64.name().toLowerCase();

  private static final String INT_32 = INT32.name().toLowerCase();

  private static final String INT_64 = DType.INT64.name().toLowerCase();

  private static final String UINT_8 = DType.UINT8.name().toLowerCase();

  private static final String BOOL = DType.BOOL.name().toLowerCase();

  private static final String STRING = DType.STRING.name().toLowerCase();

  private static final String OBJECT = DType.OBJECT.name().toLowerCase();

  private static final String UNKNOWN = DType.UNKNOWN.name().toLowerCase();

  private static final TensorType MNIST_INPUT = mnistInput();

  private static final TensorType SCALAR_TENSOR_OF_INT32 = new TensorType(INT_32, emptyList());

  private static final TensorType SCALAR_TENSOR_OF_INT64 = new TensorType(INT_64, emptyList());

  private static final TensorType SCALAR_TENSOR_OF_FLOAT32 = new TensorType(FLOAT_32, emptyList());

  private static final TensorType SCALAR_TENSOR_OF_STRING = new TensorType(STRING, emptyList());

  private static final TensorType SCALAR_TENSOR_OF_BOOL = new TensorType(BOOL, emptyList());

  private static final TensorType TENSOR_3_BOOL = TensorType.of(BOOL, 3);

  private static final TensorType TENSOR_1_1_FLOAT32 = TensorType.of(FLOAT_32, 1, 1);

  private static final TensorType TENSOR_2_3_3_FLOAT32 = TensorType.of(FLOAT_32, 2, 3, 3);

  private static final TensorType TENSOR_0_0_FLOAT32 = TensorType.of(FLOAT_32, 0, 0);

  private static final TensorType TENSOR_1_2_FLOAT32 = TensorType.of(FLOAT_32, 1, 2);

  private static final TensorType TENSOR_1_5_FLOAT32 = TensorType.of(FLOAT_32, 1, 5);

  private static final TensorType TENSOR_1_10_FLOAT32 = TensorType.of(FLOAT_32, 1, 10);

  private static final TensorType TENSOR_1_3_FLOAT32 = TensorType.of(FLOAT_32, 1, 3);

  private static final TensorType TENSOR_3_1_FLOAT32 = TensorType.of(FLOAT_32, 3, 1);

  @SuppressWarnings("unused")
  private static final TensorType TENSOR_32_INT32 = TensorType.of(INT_32, 32);

  private static final TensorType TENSOR_32_UINT8 = TensorType.of(UINT_8, 32);

  private static final TensorType TENSOR_16_UINT8 = TensorType.of(UINT_8, 16);

  private static final TensorType TENSOR_256_784_FLOAT32 = TensorType.of(FLOAT_32, 256, 784);

  private static final TensorType TENSOR_256_28_28_FLOAT32 = TensorType.of(FLOAT_32, 256, 28, 28);

  private static final TensorType TENSOR_10000_784_FLOAT32 = TensorType.of(FLOAT_32, 10000, 784);

  private static final TensorType TENSOR_5_784_FLOAT32 = TensorType.of(FLOAT_32, 5, 784);

  private static final TensorType TENSOR_60000_784_UINT8 = TensorType.of(UINT_8, 60000, 784);

  private static final TensorType TENSOR_256_10_FLOAT32 = TensorType.of(FLOAT_32, 256, 10);

  private static final TensorType TENSOR_256_UINT8 = TensorType.of(UINT_8, 256);

  private static final TensorType TENSOR_10000_10_FLOAT32 = TensorType.of(FLOAT_32, 10000, 10);

  private static final TensorType TENSOR_10000_UINT8 = TensorType.of(UINT_8, 10000);

  private static final TensorType TENSOR_32_28_28_UINT8 = TensorType.of(UINT_8, 32, 28, 28);

  private static final TensorType TENSOR_5_28_28_UINT8 = TensorType.of(UINT_8, 5, 28, 28);

  private static final TensorType TENSOR_3_28_28_UINT8 = TensorType.of(UINT_8, 3, 28, 28);

  private static final TensorType TENSOR_1_2_INT32 = TensorType.of(INT_32, 1, 2);

  private static final TensorType TENSOR_1_5_INT32 = TensorType.of(INT_32, 1, 5);

  private static final TensorType TENSOR_1_10_INT32 = TensorType.of(INT_32, 1, 10);

  private static final TensorType TENSOR_2_2_FLOAT32 = TensorType.of(FLOAT_32, 2, 2);

  private static final TensorType TENSOR_NONE_32_FLOAT32 =
      new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(32)));

  private static final TensorType TENSOR_NONE_3_FLOAT32 =
      new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(3)));

  private static final TensorType TENSOR_NONE_4_FLOAT32 =
      new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(4)));

  private static final TensorType TENSOR_NONE_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(2)));

  private static final TensorType TENSOR_NONE_NONE_STRING =
      new TensorType(STRING, asList(DynamicDim.INSTANCE, DynamicDim.INSTANCE));

  private static final TensorType TENSOR_4_RAGGED_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  private static final TensorType TENSOR_3_RAGGED_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  private static final TensorType TENSOR_4_RAGGED_RAGGED_NONE_STRING =
      new TensorType(
          STRING,
          asList(new NumericDim(4), RaggedDim.INSTANCE, RaggedDim.INSTANCE, DynamicDim.INSTANCE));

  private static final TensorType TENSOR_3_RAGGED_RAGGED_STRING =
      new TensorType(STRING, asList(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  private static final TensorType TENSOR_1_RAGGED_RAGGED_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(1), RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  private static final TensorType TENSOR_2_RAGGED_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  private static final TensorType TENSOR_2_2_INT32 = TensorType.of(INT_32, 2, 2);

  private static final TensorType TENSOR_3_2_FLOAT32 = TensorType.of(FLOAT_32, 3, 2);

  private static final TensorType TENSOR_2_4_3_FLOAT32 = TensorType.of(FLOAT_32, 2, 4, 3);

  private static final TensorType TENSOR_4_3_2_FLOAT32 = TensorType.of(FLOAT_32, 4, 3, 2);

  private static final TensorType TENSOR_4_3_FLOAT32 = TensorType.of(FLOAT_32, 4, 3);

  private static final TensorType TENSOR_2_3_FLOAT32 = TensorType.of(FLOAT_32, 2, 3);

  private static final TensorType TENSOR_1_1_3_2_FLOAT32 = TensorType.of(FLOAT_32, 1, 1, 3, 2);

  private static final TensorType TENSOR_2_3_1_FLOAT32 = TensorType.of(FLOAT_32, 2, 3, 1);

  private static final TensorType TENSOR_2_3_FLOAT64 = TensorType.of(FLOAT_64, 2, 3);

  private static final TensorType TENSOR_4_INT64 = TensorType.of(INT_64, 4);

  private static final TensorType TENSOR_100_784_FLOAT32 = TensorType.of(FLOAT_32, 100, 784);

  private static final TensorType TENSOR_4_8_FLOAT32 = TensorType.of(FLOAT_32, 4, 8);

  private static final TensorType TENSOR_4_512_FLOAT32 = TensorType.of(FLOAT_32, 4, 512);

  private static final TensorType TENSOR_2_64_FLOAT32 = TensorType.of(FLOAT_32, 2, 64);

  private static final TensorType SPARSE_TENSOR_4_4_FLOAT32 =
      new SparseTensorType(FLOAT32, asList(new NumericDim(4), new NumericDim(4)));

  private static final TensorType TENSOR_4_10_FLOAT32 = TensorType.of(FLOAT_32, 4, 10);

  private static final TensorType TENSOR_4_1_INT32 = TensorType.of(INT_32, 4, 1);

  private static final TensorType TENSOR_256_256_3_FLOAT32 = TensorType.of(FLOAT_32, 256, 256, 3);

  private static final TensorType TENSOR_2_3_4_FLOAT32 = TensorType.of(FLOAT_32, 2, 3, 4);

  private static final TensorType TENSOR_2_2_4_FLOAT32 = TensorType.of(FLOAT_32, 2, 2, 4);

  private static final TensorType TENSOR_2_2_1_FLOAT32 = TensorType.of(FLOAT_32, 2, 2, 1);

  private static final TensorType TENSOR_2_5_6_FLOAT32 = TensorType.of(FLOAT_32, 2, 5, 6);

  private static final TensorType TENSOR_4_4_6_FLOAT32 = TensorType.of(FLOAT_32, 4, 4, 6);

  private static final TensorType TENSOR_4_6_FLOAT32 = TensorType.of(FLOAT_32, 4, 6);

  private static final TensorType TENSOR_1_2_2_27_FLOAT32 = TensorType.of(FLOAT_32, 1, 2, 2, 27);

  private static final TensorType TENSOR_4_4_FLOAT32 = TensorType.of(FLOAT_32, 4, 4);

  private static final TensorType TENSOR_2_5_INT32 = TensorType.of(INT_32, 2, 5);

  private static final TensorType TENSOR_3_3_FLOAT32 = TensorType.of(FLOAT_32, 3, 3);

  private static final TensorType TENSOR_3_3_INT32 = TensorType.of(INT_32, 3, 3);

  private static final TensorType TENSOR_0_RAGGED_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(0), RaggedDim.INSTANCE));

  private static final TensorType TENSOR_0_RAGGED_3_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(0), RaggedDim.INSTANCE, new NumericDim(3)));

  private static final TensorType TENSOR_1_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(1), RaggedDim.INSTANCE));

  private static final TensorType TENSOR_1_RAGGED_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(1), RaggedDim.INSTANCE));

  private static final TensorType TENSOR_2_NONE_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), DynamicDim.INSTANCE));

  private static final TensorType TENSOR_2_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), RaggedDim.INSTANCE));

  private static final TensorType TENSOR_NONE_RAGGED_INT32 =
      new TensorType(INT_32, asList(DynamicDim.INSTANCE, RaggedDim.INSTANCE));

  private static final TensorType TENSOR_2_RAGGED_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(2), RaggedDim.INSTANCE));

  private static final TensorType TENSOR_2_RAGGED_2_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(2), RaggedDim.INSTANCE, new NumericDim(2)));

  private static final TensorType TENSOR_2_RAGGED_2_INT32 =
      new TensorType(INT_32, asList(new NumericDim(2), RaggedDim.INSTANCE, new NumericDim(2)));

  private static final TensorType TENSOR_2_RAGGED_2_3_INT32 =
      new TensorType(
          INT_32,
          asList(new NumericDim(2), RaggedDim.INSTANCE, new NumericDim(2), new NumericDim(3)));

  private static final TensorType TENSOR_2_RAGGED_2_2_INT32 =
      new TensorType(
          INT_32,
          asList(new NumericDim(2), RaggedDim.INSTANCE, new NumericDim(2), new NumericDim(2)));

  private static final TensorType TENSOR_2_RAGGED_RAGGED_RAGGED_FLOAT32 =
      new TensorType(
          FLOAT_32,
          asList(new NumericDim(2), RaggedDim.INSTANCE, RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  private static final TensorType TENSOR_3_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(3), RaggedDim.INSTANCE));

  private static final TensorType TENSOR_3_RAGGED_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), RaggedDim.INSTANCE));

  private static final TensorType TENSOR_4_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(4), RaggedDim.INSTANCE));

  private static final TensorType TENSOR_3_RAGGED_RAGGED_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), RaggedDim.INSTANCE, RaggedDim.INSTANCE));

  private static final TensorType TENSOR_3_RAGGED_1_FLOAT32 =
      new TensorType(FLOAT_32, asList(new NumericDim(3), RaggedDim.INSTANCE, new NumericDim(1)));

  private static final TensorType TENSOR_2_3_INT32 = TensorType.of(INT_32, 2, 3);

  private static final TensorType TENSOR_2_4_FLOAT32 = TensorType.of(FLOAT_32, 2, 4);

  private static final TensorType TENSOR_2_6_INT32 = TensorType.of(INT_32, 2, 6);

  private static final TensorType TENSOR_2_1_FLOAT32 = TensorType.of(FLOAT_32, 2, 1);

  private static final TensorType TENSOR_10_2_FLOAT32 = TensorType.of(FLOAT_32, 10, 2);

  private static final TensorType TENSOR_10_2_FLOAT64 = TensorType.of(FLOAT_64, 10, 2);

  private static final TensorType TENSOR_5_2_FLOAT32 = TensorType.of(FLOAT_32, 5, 2);

  private static final TensorType TENSOR_5_2_INT32 = TensorType.of(INT_32, 5, 2);

  private static final TensorType TENSOR_5_5_FLOAT32 = TensorType.of(FLOAT_32, 5, 5);

  private static final TensorType TENSOR_5_5_INT32 = TensorType.of(INT_32, 5, 5);

  private static final TensorType TENSOR_5_RAGGED_INT32 =
      new TensorType(INT_32, asList(new NumericDim(5), RaggedDim.INSTANCE));

  private static final TensorType TENSOR_2_3_3_INT32 = TensorType.of(INT_32, 2, 3, 3);

  private static final TensorType TENSOR_2_3_4_INT32 = TensorType.of(INT_32, 2, 3, 4);

  private static final TensorType TENSOR_2_5_3_FLOAT32 = TensorType.of(FLOAT_32, 2, 5, 3);

  private static final TensorType TENSOR_3_2_2_FLOAT32 = TensorType.of(FLOAT_32, 3, 2, 2);

  private static final TensorType TENSOR_5_6_FLOAT32 = TensorType.of(FLOAT_32, 5, 6);

  private static final TensorType TENSOR_30_FLOAT32 = TensorType.of(FLOAT_32, 30);

  private static final TensorType TENSOR_4_5_6_FLOAT32 = TensorType.of(FLOAT_32, 4, 5, 6);

  private static final TensorType TENSOR_7_5_2_FLOAT32 = TensorType.of(FLOAT_32, 7, 5, 2);

  private static final TensorType TENSOR_30_3_2_FLOAT32 = TensorType.of(FLOAT_32, 30, 3, 2);

  private static final TensorType TENSOR_3_2_2_3_FLOAT32 = TensorType.of(FLOAT_32, 3, 2, 2, 3);

  private static final TensorType TENSOR_2_2_2_3_FLOAT32 = TensorType.of(FLOAT_32, 2, 2, 2, 3);

  private static final TensorType TENSOR_20_28_28_FLOAT32 = TensorType.of(FLOAT_32, 20, 28, 28);

  private static final TensorType TENSOR_20_28_28_INT32 = TensorType.of(INT_32, 20, 28, 28);

  private static final TensorType TENSOR_20_10_FLOAT32 = TensorType.of(FLOAT_32, 20, 10);

  private static final TensorType TENSOR_20_64_FLOAT32 = TensorType.of(FLOAT_32, 20, 64);

  private static final TensorType TENSOR_60000_28_28_FLOAT32 =
      TensorType.of(FLOAT_32, 60000, 28, 28);

  private static final TensorType TENSOR_60000_28_28_UINT8 = TensorType.of(UINT_8, 60000, 28, 28);

  private static final TensorType TENSOR_50000_32_32_3_UINT8 =
      TensorType.of(UINT_8, 50000, 32, 32, 3);

  private static final TensorType TENSOR_8982_INT64 = TensorType.of(INT_64, 8982);

  private static final TensorType TENSOR_404_13_FLOAT64 = TensorType.of(FLOAT_64, 404, 13);

  private static final TensorType TENSOR_404_FLOAT64 = TensorType.of(FLOAT_64, 404);

  private static final TensorType TENSOR_60000_UINT8 = TensorType.of(UINT_8, 60000);

  private static final TensorType TENSOR_50000_1_UINT8 = TensorType.of(UINT_8, 50000, 1);

  private static final TensorType TENSOR_50000_1_INT64 = TensorType.of(INT_64, 50000, 1);

  private static final TensorType TENSOR_8982_OBJECT = TensorType.of(OBJECT, 8982);

  private static final TensorType TENSOR_102_13_FLOAT64 = TensorType.of(FLOAT_64, 102, 13);

  private static final TensorType TENSOR_102_FLOAT64 = TensorType.of(FLOAT_64, 102);

  private static final TensorType TENSOR_10000_32_32_3_UINT8 =
      TensorType.of(UINT_8, 10000, 32, 32, 3);

  private static final TensorType TENSOR_10000_1_UINT8 = TensorType.of(UINT_8, 10000, 1);

  private static final TensorType TENSOR_10000_1_INT64 = TensorType.of(INT_64, 10000, 1);

  private static final TensorType TENSOR_10000_28_28_UINT8 = TensorType.of(UINT_8, 10000, 28, 28);

  private static final TensorType TENSOR_2246_INT64 = TensorType.of(INT_64, 2246);

  private static final TensorType TENSOR_2246_OBJECT = TensorType.of(OBJECT, 2246);

  /** A {@code float32} tensor whose shape cannot be statically inferred. */
  private static final TensorType TENSOR_UNKNOWN_SHAPE_FLOAT32 = new TensorType(FLOAT_32, null);

  /** Fully-⊤ tensor type: unknown shape and unknown dtype. */
  private static final TensorType TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE =
      new TensorType(UNKNOWN, null);

  private static final TensorType TENSOR_1_FLOAT32 = TensorType.of(FLOAT_32, 1);

  private static final TensorType TENSOR_2_FLOAT32 = TensorType.of(FLOAT_32, 2);

  private static final TensorType TENSOR_2_FLOAT64 = TensorType.of(FLOAT_64, 2);

  private static final TensorType TENSOR_DYNAMIC_DYNAMIC_FLOAT32 =
      new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, DynamicDim.INSTANCE));

  private static final TensorType TENSOR_2_INT32 = TensorType.of(INT_32, 2);

  private static final TensorType TENSOR_2_INT64 = TensorType.of(INT_64, 2);

  private static final TensorType TENSOR_INT64_UNKNOWN_SHAPE = new TensorType(INT_64, null);

  private static final TensorType TENSOR_DYNAMIC_INT64 =
      new TensorType(INT_64, asList(DynamicDim.INSTANCE));

  private static final TensorType TENSOR_INT32_UNKNOWN_SHAPE = new TensorType(INT_32, null);

  private static final TensorType TENSOR_1_0_0_9_INT32 = TensorType.of(INT_32, 1, 0, 0, 9);

  private static final TensorType TENSOR_UNKNOWN_SHAPE_BOOL = new TensorType(BOOL, null);

  private static final TensorType TENSOR_3_INT32 = TensorType.of(INT_32, 3);

  private static final TensorType TENSOR_3_INT64 = TensorType.of(INT_64, 3);

  private static final TensorType TENSOR_3_FLOAT32 = TensorType.of(FLOAT_32, 3);

  private static final TensorType TENSOR_4_FLOAT32 = TensorType.of(FLOAT_32, 4);

  private static final TensorType TENSOR_2_2_BOOL = TensorType.of(BOOL, 2, 2);

  private static final TensorType TENSOR_3_5_BOOL = TensorType.of(BOOL, 3, 5);

  private static final TensorType TENSOR_3_5_INT32 = TensorType.of(INT_32, 3, 5);

  private static final TensorType TENSOR_3_5_FLOAT32 = TensorType.of(FLOAT_32, 3, 5);

  private static final TensorType TENSOR_4_FLOAT64 = TensorType.of(FLOAT_64, 4);

  private static final TensorType TENSOR_5_FLOAT32 = TensorType.of(FLOAT_32, 5);

  private static final TensorType TENSOR_5_FLOAT64 = TensorType.of(FLOAT_64, 5);

  private static final TensorType TENSOR_64_5_FLOAT32 = TensorType.of(FLOAT_32, 64, 5);

  private static final TensorType TENSOR_7_FLOAT32 = TensorType.of(FLOAT_32, 7);

  private static final TensorType TENSOR_32_7_FLOAT32 = TensorType.of(FLOAT_32, 32, 7);

  private static final TensorType TENSOR_64_7_FLOAT32 = TensorType.of(FLOAT_32, 64, 7);

  private static final TensorType TENSOR_20_5_FLOAT32 = TensorType.of(FLOAT_32, 20, 5);

  private static final TensorType TENSOR_20_7_FLOAT32 = TensorType.of(FLOAT_32, 20, 7);

  private static final TensorType TENSOR_5_INT32 = TensorType.of(INT_32, 5);

  private static final TensorType TENSOR_5_INT64 = TensorType.of(INT_64, 5);

  private static final TensorType TENSOR_4_INT32 = TensorType.of(INT_32, 4);

  private static final TensorType TENSOR_1_INT32 = TensorType.of(INT_32, 1);

  private static final TensorType TENSOR_3_4_INT32 = TensorType.of(INT_32, 3, 4);

  private static final TensorType TENSOR_3_4_FLOAT32 = TensorType.of(FLOAT_32, 3, 4);

  private static final TensorType TENSOR_4_5_FLOAT32 = TensorType.of(FLOAT_32, 4, 5);

  private static final TensorType TENSOR_1_28_28_1_FLOAT32 = TensorType.of(FLOAT_32, 1, 28, 28, 1);

  private static final TensorType TENSOR_6_INT32 = TensorType.of(INT_32, 6);

  private static final TensorType TENSOR_6_FLOAT32 = TensorType.of(FLOAT_32, 6);

  private static final TensorType TENSOR_256_28_28_1_FLOAT32 =
      TensorType.of(FLOAT_32, 256, 28, 28, 1);

  private static final TensorType TENSOR_32_28_28_1_FLOAT32 =
      TensorType.of(FLOAT_32, 32, 28, 28, 1);

  private static final TensorType TENSOR_16_28_28_1_FLOAT32 =
      TensorType.of(FLOAT_32, 16, 28, 28, 1);

  private static final TensorType TENSOR_256_64_FLOAT32 = TensorType.of(FLOAT_32, 256, 64);

  private static final TensorType TENSOR_96_28_28_1_FLOAT32 =
      TensorType.of(FLOAT_32, 96, 28, 28, 1);

  private static final TensorType TENSOR_4096_32_32_3_FLOAT32 =
      TensorType.of(FLOAT_32, 4096, 32, 32, 3);

  private static final TensorType TENSOR_4096_UINT8 = TensorType.of(UINT_8, 4096);

  private static final TensorType TENSOR_3_STRING = TensorType.of(STRING, 3);

  private static final TensorType TENSOR_25000_INT64 = TensorType.of(INT_64, 25000);

  private static final TensorType TENSOR_25000_OBJECT = TensorType.of(OBJECT, 25000);

  @Test
  public void testValueIndex()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_value_index.py",
        "value_index",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32), 3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testValueIndex2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_value_index2.py",
        "value_index",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32), 3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testValueIndex3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_value_index3.py",
        "value_index",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32), 3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testValueIndex4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_value_index4.py",
        "value_index",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32), 3, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testFunction()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function2.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function3.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function4.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function5.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testFunction6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function6.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_1_FLOAT32)));
  }

  @Test
  public void testFunction7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function7.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testFunction8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_function8.py",
        "func",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_1_FLOAT32, TENSOR_2_FLOAT32)));
  }

  @Test
  public void testFunction9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function9.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testFunction10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function10.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_INT32)));
  }

  @Test
  public void testFunction11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_function11.py", "func", 1, 1, Map.of(2, Set.of(TENSOR_2_3_3_INT32)));
  }

  /** Test https://github.com/wala/ML/issues/308. */
  @Test
  public void testFunction12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_function12.py",
        "func",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_1_FLOAT32, TENSOR_3_2_FLOAT32)));
  }

  /**
   * Test https://github.com/wala/ML/issues/308.
   *
   * <p>This one has lexical scoping.
   */
  @Test
  public void testFunction13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_function13.py",
        "func",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_1_FLOAT32, TENSOR_3_2_FLOAT32)));
  }

  @Test
  public void testDecorator()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator2.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator3.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testDecorator4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator4.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator5.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator6.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator7.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator8.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator9.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator10.py", "returned", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testDecorator11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorator11.py", "C.returned", 1, 1, Map.of(3, Set.of(TENSOR_5_INT32)));
  }

  /**
   * The {@code returned(a)} parameter is {@code a = tf.constant([1, 1.0])}, i.e. {@code (2,)
   * float32}. The asserted set is a union across contexts (per the test helper's union-per-vn
   * semantics): the {@code (2,) float32} is the parameter's real type, now precise after the top_k
   * output-shape composer (<a href="https://github.com/wala/ML/issues/609">wala/ML#609</a>)
   * sharpened the previous ⊤. The {@code (2,) int32} is a top_k {@code indices} output leaking in
   * through a collapsed 1-CFA context (it was already present in the old union as ⊤ int32); the
   * composer only made it concrete. The leak itself is a context-sensitivity artifact.
   */
  @Test
  public void testDecorator12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_decorator12.py",
        "returned",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_FLOAT32, TENSOR_2_INT32)));
  }

  @Test
  public void testDataset()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDataset2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset2.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Regression guard for wala/ML#548: a {@code tf.reshape} on the path to a {@code @tf.function}
   * parameter must not degrade the inferred parameter. Mirrors the cifar10 path in main.py ({@code
   * tf.reshape(labels)} feeding a {@code tf.data.Dataset} whose iteration binds {@code labels});
   * {@code consume(labels)} pins the type. The parameter is concrete and dtype-exact: {@code uint8}
   * (matching cifar10's label dtype), and the union carries both correct batch shapes &mdash;
   * {@code (32,)} for the full batches and {@code (16,)} for the final partial batch (cifar10's
   * 50000 train labels batched by 32 leave {@code 50000 % 32 == 16}, since {@code drop_remainder}
   * defaults to false). So the reshape not only fails to degrade the parameter, the parameter is
   * inferred precisely. See ponder-lab/Input-Signature-Inference-Paper#49.
   */
  @Test
  public void testReshapeToParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_to_param.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_32_UINT8, TENSOR_16_UINT8)));
  }

  /** This is not a legal case. */
  @Test
  public void testDataset3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset3.py", "add", 0, 0, emptyMap());
  }

  /** This is not a legal case. */
  @Test
  public void testDataset4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset4.py", "add", 0, 0, emptyMap());
  }

  @Test
  public void testDataset5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset5.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  @Test
  public void testDataset6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset6.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  @Test
  public void testDataset7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset7.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  @Test
  public void testDataset8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset8.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_32_28_28_UINT8), 3, Set.of(TENSOR_32_UINT8)));
  }

  @Test
  public void testDataset9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset9.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_32_28_28_UINT8), 3, Set.of(TENSOR_32_UINT8)));
  }

  @Test
  public void testDataset10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset10.py",
        "add",
        2,
        3,
        Map.of(
            2,
            Set.of(SCALAR_TENSOR_OF_INT32, TENSOR_2_NONE_INT32),
            3,
            Set.of(SCALAR_TENSOR_OF_INT32, TENSOR_2_NONE_INT32)));
  }

  @Test
  public void testDataset10a()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset10a.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(SCALAR_TENSOR_OF_INT32),
            3, Set.of(TENSOR_2_NONE_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset11.py", "f", 0, 0);
    test("tf2_test_dataset11.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset12.py", "f", 0, 0);
    test("tf2_test_dataset12.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset13.py", "f", 0, 0);
    test("tf2_test_dataset13.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset14()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset14.py", "f", 0, 0);
    test("tf2_test_dataset14.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset15()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset15.py", "f", 0, 0);
    test("tf2_test_dataset15.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset16()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset16.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset17()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset17.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset17.py", "f", 1, 2, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset18()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset18.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset18.py", "f", 1, 2, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset18.py", "g", 0, 2);
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset19()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType images = TensorType.of(FLOAT_32, 512, 112, 112, 3);
    TensorType labels = new TensorType(FLOAT_32, asList(new NumericDim(512), DynamicDim.INSTANCE));

    test(
        "tf2_test_dataset19.py", "distributed_train_step", 1, 1, Map.of(2, Set.of(images, labels)));
  }

  @Test
  public void testDataset20()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset20.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset21()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset21.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset22()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset22.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset23()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset23.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset23.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDataset24()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset24.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset24.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset25()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset25.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset25.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/165 is fixed.
    test(
        "tf2_test_dataset25.py",
        "h",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64, SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDataset26()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset26.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset26.py", "g1", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset26.py", "g2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset26.py", "g3", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/165 is fixed.
    test(
        "tf2_test_dataset26.py",
        "h",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64, TENSOR_2_INT32)));
  }

  @Test
  public void testDataset27()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset27.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
    test("tf2_test_dataset27.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset27.py", "h", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset27.py", "i", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDataset28()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset28.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
    test("tf2_test_dataset28.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
    // TODO: Change to 0, 0 when https://github.com/wala/ML/issues/164 is fixed:
    test(
        "tf2_test_dataset28.py",
        "h",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_INT32, SCALAR_TENSOR_OF_STRING)));
  }

  @Test
  public void testDataset29()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset29.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  /**
   * Regression test for wala/ML#452: iterating a {@code tf.data.TextLineDataset} via {@code for
   * element in dataset:} should classify each element as a 0-D string tensor. The receiving
   * function {@code func}'s parameter at {@code vn=2} must therefore have type {@code
   * SCALAR_TENSOR_OF_STRING}. Pre-fix, the analyzer's {@code TextLineDataset} model didn't preserve
   * the per-element tensor type through the iteration substrate, leaving {@code func} with no
   * tensor classification at all (downstream {@code Function.getHasTensorParameter()} reported
   * false).
   */
  @Test
  public void testTextLineDatasetIterationElementType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_textlinedataset_iter.py",
        "func",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  @Test
  public void testDataset30()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset30.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset31()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/166 is fixed.
    test("tf2_test_dataset31.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "g1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "g2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/166 is fixed.
    test("tf2_test_dataset31.py", "h", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "i1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "i2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/166 is fixed.
    test(
        "tf2_test_dataset31.py",
        "j",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64, TENSOR_2_INT64)));
    test("tf2_test_dataset31.py", "k1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "k2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "k3", 1, 1, Map.of(2, Set.of(TENSOR_2_INT64)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/166 is fixed.
    test("tf2_test_dataset31.py", "l", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "m1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "m2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset32()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset32.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset33()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset33.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset34()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset34.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset35()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset35.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset36()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset36.py", "id1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
    //    test("tf2_test_dataset36.py", "id2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset37()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset37.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case that exercises a potential 1-CFA precision problem with Datasets by using a shared
   * function to retrieve the first element from two different datasets.
   */
  @Test
  public void testDataset38()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset38.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset38.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Control test for {@link #testDataset38()}, utilizing only a single dataset to ensure the 1-CFA
   * precision issue is specifically due to the merging of multiple datasets.
   */
  @Test
  public void testDataset39()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset39.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case that exercises a potential 1-CFA precision problem with Datasets by using a shared
   * function to retrieve the first element from two different datasets. Uses tf.constant to bypass
   * unrelated type inference issues with Python lists in from_tensor_slices.
   */
  @Test
  public void testDataset40()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset40.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset40.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Control test for {@link #testDataset40()}, utilizing only a single dataset to ensure the 1-CFA
   * precision issue is specifically due to the merging of multiple datasets.
   */
  @Test
  public void testDataset41()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset41.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Direct access test based on {@link #testDataset38()}, iterating over the dataset directly
   * instead of through a shared function, avoiding the 1-CFA precision issue.
   */
  @Test
  public void testDataset42()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset42.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset42.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset42()}, utilizing only a single dataset. */
  @Test
  public void testDataset43()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset43.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Direct access test based on {@link #testDataset40()}, using tf.constant and iterating directly.
   */
  @Test
  public void testDataset44()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset44.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset44.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset44()}, utilizing only a single dataset. */
  @Test
  public void testDataset45()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset45.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case based on {@link #testDataset38()}, exercising potential 1-CFA precision problem with
   * Datasets by varying the shape (e.g., shape (2,) vs (2, 2)) instead of the dtype.
   */
  @Test
  public void testDataset46()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset46.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset46.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset46()}, utilizing only a single dataset. */
  @Test
  public void testDataset47()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset47.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Test case based on {@link #testDataset40()}, exercising potential 1-CFA precision problem with
   * Datasets by varying the shape using tf.constant.
   */
  @Test
  public void testDataset48()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset48.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset48.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset48()}, utilizing only a single dataset. */
  @Test
  public void testDataset49()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset49.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Direct access test based on {@link #testDataset46()}, iterating over the dataset directly
   * instead of through a shared function.
   */
  @Test
  public void testDataset50()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset50.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset50.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset50()}, utilizing only a single dataset. */
  @Test
  public void testDataset51()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset51.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Direct access test based on {@link #testDataset48()}, using tf.constant and iterating directly.
   */
  @Test
  public void testDataset52()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset52.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset52.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset52()}, utilizing only a single dataset. */
  @Test
  public void testDataset53()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset53.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Test case that exercises a potential 1-CFA precision problem with Datasets by using an explicit
   * iterator inside a shared function to retrieve the first element from two different datasets.
   */
  @Test
  public void testDataset54()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset54.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset54.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset54()}, utilizing only a single dataset. */
  @Test
  public void testDataset55()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset55.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case that exercises a potential 1-CFA precision problem with chained Datasets by sharing a
   * helper function that calls shuffle() on two different datasets.
   */
  @Test
  public void testDataset56()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset56.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset56.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset56()}, utilizing only a single dataset. */
  @Test
  public void testDataset57()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset57.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Direct access version of {@link #testDataset56()}, utilizing chained Datasets (shuffle)
   * directly in the script without a shared helper function. This should NOT suffer from 1-CFA
   * precision merging.
   */
  @Test
  public void testDataset58()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset58.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset58.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset58()}, utilizing only a single dataset. */
  @Test
  public void testDataset59()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset59.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Direct access version of {@link #testDataset54()}, utilizing an explicit iterator directly in
   * the script without a shared helper function.
   */
  @Test
  public void testDataset60()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset60.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset60.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset60()}, utilizing only a single dataset. */
  @Test
  public void testDataset61()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset61.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case exposing 1-CFA precision problem using {@code tf.data.Dataset.from_tensors} inside a
   * shared wrapper function.
   */
  @Test
  public void testDataset62()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset62.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
    test("tf2_test_dataset62.py", "g", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3))));
  }

  /** Control test for {@link #testDataset62()}, utilizing only a single dataset. */
  @Test
  public void testDataset63()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset63.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * Test case exposing 1-CFA precision problem using {@code tf.data.Dataset.range} inside a shared
   * wrapper function.
   */
  @Test
  public void testDataset64()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // The range generator always produces int64, but the 1-CFA merge occurs anyway,
    // though the type sets will both be {int64}. We verify it runs.
    test("tf2_test_dataset64.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset64.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  /** Control test for {@link #testDataset64()}, utilizing only a single dataset. */
  @Test
  public void testDataset65()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset65.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  /**
   * Test case exposing 1-CFA precision problem using {@code shuffle} inside a shared wrapper
   * function, varying shapes instead of dtypes.
   */
  @Test
  public void testDataset66()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset66.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset66.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset66()}, utilizing only a single dataset. */
  @Test
  public void testDataset67()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset67.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Test case exposing 1-CFA precision problem using {@code tf.data.Dataset.from_tensors} inside a
   * shared wrapper function, varying shapes instead of dtypes.
   */
  @Test
  public void testDataset68()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset68.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset68.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset68()}, utilizing only a single dataset. */
  @Test
  public void testDataset69()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset69.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Test {@code tf.data.Dataset.from_tensor_slices} with a tuple of two Python list literals,
   * followed by the same {@code .shuffle(...).batch(...)} chain as {@link #testDataset5()}. The
   * iterator yields a tuple {@code (element_a, element_b)} where each element has the same batched
   * shape as {@code testDataset5}'s single element. This isolates the "tuple-structured argument"
   * path from the {@link #testDataset5()} "single-list" path; every test that feeds a tuple to
   * {@code from_tensor_slices} today does so via {@code mnist.load_data()}'s ndarray split, and no
   * existing test exercises the tuple-of-literals case in isolation.
   *
   * <p>The root-cause tuple walk was fixed on the feature branch for wala/ML#366 via a call-site
   * helper in {@link com.ibm.wala.cast.python.ml.client.DatasetFromTensorSlicesGenerator}, but this
   * test still fails because of a separate "chain-preservation gap": the {@code .shuffle().batch()}
   * chain wraps the dataset in generators that don't implement {@code DelegatingTensorGenerator},
   * so the factory's {@code PythonPropertyRead} dispatch for the {@code for element_a, element_b in
   * dataset:} unpack cannot peel back to find the tuple structure, and instead wraps in {@code
   * TensorElementGenerator} which over-peels one dim — collapsing the batched shape {@code (2,)}
   * back to {@code ()}.
   */
  @Test
  public void testDataset70()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset70.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  /**
   * Like {@link #testDataset70()} but the two elements of the tuple are {@code tf.constant(...)}
   * calls rather than raw Python list literals. Used as a comparison point to isolate whether the
   * tuple-structured-argument bug observed in {@link #testDataset70()} is specific to raw literals
   * or also applies when the tuple elements are already typed tensors. Both fail with identical
   * wrong output; see {@link #testDataset70()}'s Javadoc for the current root-cause status
   * (wala/ML#366 root-cause walk fixed, chain-preservation gap remains).
   */
  @Test
  public void testDataset71()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset71.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  /**
   * Test a dataset created with {@code tf.data.Dataset.from_generator} using a <em>dict</em>
   * -structured legacy {@code output_types} argument (https://github.com/wala/ML/issues/615). The
   * dtype specification is a mapping, not a scalar {@code tf.DType} or a tuple/list of them, so the
   * dtype helper must recurse into the dict's values rather than asserting a single {@code DType}.
   */
  @Test
  public void testDataset72()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset72.py", "consume", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   *
   * <p>{@code summarize_weights(step)} takes only an {@code int} step counter (not a tensor, hence
   * parameter count 0). Inside the function, dict lookups {@code weights[w]} and {@code biases[b]}
   * feed 4 tensor variables to {@code tf.summary.histogram}. Previously this test registered 5: the
   * extra was a spurious {@code v2} (the {@code step} parameter) typed {@code {(256, 784) float32}}
   * via PA-graph propagation from the {@code enumerate()} call at the caller ({@code for step, (x,
   * y) in enumerate(ds, 1):}). The leak was at the PA substrate &mdash; tensor types flowing to the
   * enumerate tuple's field-0 slot through the PTS-graph edge, not at generator dispatch.
   *
   * <p>Closed by wala/ML#409: {@link com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine}
   * now detects enumerate-first-field reads structurally and pins their {@link
   * com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis} state to empty via a {@code DropOp}
   * edge transfer that both clears existing leaked state and FIXes the slot against further
   * predecessor updates. Count correctly reports 4.
   */
  @Test
  public void testTensorboardExample()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tensorboard_example.py", "summarize_weights", 0, 4);
  }

  @Test
  public void testTensorList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tensor_list.py",
        "add",
        2,
        3,
        Map.of(
            2,
            Set.of(TENSOR_1_2_FLOAT32, TENSOR_2_2_FLOAT32),
            3,
            Set.of(TENSOR_1_2_FLOAT32, TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testTensorList2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensor_list2.py", "add", 0, 1);
  }

  @Test
  public void testTensorList3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // `list.append` is modeled (wala/ML#136): both appended tensors reach `add` through the
    // iteration, so each parameter carries the union of the two shapes, as in testTensorList.
    test(
        "tf2_test_tensor_list3.py",
        "add",
        2,
        3,
        Map.of(
            2,
            Set.of(TENSOR_1_2_FLOAT32, TENSOR_2_2_FLOAT32),
            3,
            Set.of(TENSOR_1_2_FLOAT32, TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testTensorList4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensor_list4.py", "add", 0, 0);
  }

  @Test
  public void testTensorList5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensor_list5.py", "add", 0, 0);
  }

  /**
   * Six tensor variables in {@code SequentialModel.__call__}: the {@code x} parameter (vn=3, shape
   * {@code (20, 28, 28) f32}) plus five intermediate SSA values produced by the {@code
   * self.flatten(x) → 100× Dense(64) → self.dropout(x) → self.dense_2(x)} chain. The {@code
   * Flatten} result is concrete {@code (20, 784)} via {@link
   * com.ibm.wala.cast.python.ml.client.FlattenCall} (vn=9); the loop's {@code Dense(64)} output is
   * concrete {@code (20, 64)} (vn=22); the loop-head phi (vn=17) and the {@code Dropout} output
   * (vn=26) union both reachable shapes {@code {(20, 784), (20, 64)}} (the pre-loop entry shape
   * survives because an empty {@code my_layers} would leave {@code x} at the {@code Flatten}
   * shape); the final {@code Dense(10)} output is concrete {@code (20, 10)} (vn=30).
   *
   * <p>The loop's {@code Dense(64)} narrows because {@code range(n)} now returns an iterable
   * (non-empty) list, so the {@code self.my_layers} comprehension populates the list and the {@code
   * self.my_layers[idx]} subscript read resolves to its {@code Dense(64)} elements (<a
   * href="https://github.com/wala/ML/issues/599">wala/ML#599</a>). The direct {@code Dense(10)}
   * narrows via the SSA-chain fallback in {@link
   * com.ibm.wala.cast.python.ml.client.DenseCall#getDefaultShapes} (<a
   * href="https://github.com/wala/ML/issues/358">wala/ML#358</a>).
   */
  @Test
  public void testModelCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call.py",
        "SequentialModel.__call__",
        1,
        6,
        Map.of(3, Set.of(TENSOR_20_28_28_FLOAT32)));
  }

  @Test
  public void testModelCall2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call2.py",
        "SequentialModel.call",
        1,
        6,
        Map.of(3, Set.of(TENSOR_20_28_28_FLOAT32)));
  }

  @Test
  public void testModelCall3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call3.py",
        "SequentialModel.call",
        1,
        6,
        Map.of(3, Set.of(TENSOR_20_28_28_FLOAT32)));
  }

  @Test
  public void testModelCall4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call4.py",
        "SequentialModel.__call__",
        1,
        6,
        Map.of(3, Set.of(TENSOR_20_28_28_FLOAT32)));
  }

  /**
   * Test call string imprecision as described in
   * https://github.com/wala/WALA/discussions/1417#discussioncomment-10085680. This should fail due
   * to https://github.com/wala/ML/issues/207.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#207 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testModelCall5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "proj66/src/tf2_test_model_call5b.py",
          "proj66/tf2_test_model_call5.py",
          "proj66/tf2_test_model_call5a.py"
        },
        "tf2_test_model_call5.py",
        "SequentialModel.__call__",
        "proj66",
        1,
        1,
        Map.of(3, Set.of(MNIST_INPUT)));

    test(
        new String[] {
          "proj66/src/tf2_test_model_call5b.py",
          "proj66/tf2_test_model_call5.py",
          "proj66/tf2_test_model_call5a.py"
        },
        "tf2_test_model_call5a.py",
        "SequentialModel.__call__",
        "proj66",
        1,
        1,
        Map.of(3, Set.of(MNIST_INPUT)));
  }

  /**
   * Test https://github.com/wala/ML/issues/267.
   *
   * <p>Explicit dtype.
   */
  @Test
  public void testModelCall6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call6.py",
        "SequentialModel.__call__",
        1,
        6,
        Map.of(3, Set.of(TENSOR_20_28_28_INT32)));
  }

  /**
   * Regression guard for wala/ML#655: a {@code tf.io.FixedLenFeature} value, parsed by {@code
   * tf.io.parse_single_example} and read back through a dict subscript, types as a dense tensor
   * whose shape comes from the feature's {@code dims} argument and whose dtype comes from its
   * {@code type} argument. Previously {@code FixedLenFeature.do} allocated an {@code
   * Ltensorflow/objects/ feature} that the manual tensor walker ignores, so the parsed value never
   * typed.
   */
  @Test
  public void testFixedLenFeature()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_fixed_len_feature.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 128))));
  }

  /**
   * Regression guard for wala/ML#655, end to end: the NLPGNN {@code TFLoader.load_valid} input
   * pipeline — a {@code TFRecordDataset} mapped by a {@code parse_single_example} decoder returning
   * a 4-tuple of {@code FixedLenFeature} dict-subscripts, then {@code prefetch}ed, then iterated
   * with a 4-way tuple unpack — types its first parsed field {@code X} ({@code input_ids}) to
   * {@code (128,)} int64. Before the {@code FixedLenFeature} fix, the mapped element was
   * non-tensor, so {@code X} (and any model parameter fed from it) came back non-tensor.
   */
  @Test
  public void testTfrecordLoader()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tfrecord_loader.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 128))));
  }

  /**
   * Regression guard for wala/ML#655, the full symptom-A chain in one fixture: the NLPGNN {@code
   * BilstmAttention} model (whose {@code predict} forwards to {@code self(inputs, training)} and
   * whose {@code call} delegates to a user-defined child {@code BiLSTM} layer built from unmodeled
   * sublayers) fed from the {@code TFLoader} {@code FixedLenFeature}/{@code TFRecordDataset}
   * source. The child {@code BiLSTM.call}'s {@code inputs} parameter types to {@code (128,)} int64,
   * flowing the parsed {@code input_ids} field through {@code model.predict(X)} → {@code __call__}
   * → {@code call} → {@code self.bilstm(inputs, training)}. This demonstrates that the symptom-A
   * cases are unit-reproducible via the source pipeline (not the affected file, which carries no
   * call site or source) and that the cause was the source typing, not the {@code __call__}
   * forwarding the issue title hypothesized. Before the {@code FixedLenFeature} fix, {@code inputs}
   * came back non-tensor.
   */
  @Test
  public void testBilstmLoaderE2e()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_bilstm_loader_e2e.py",
        "BiLSTM.call",
        1,
        2,
        Map.of(3, Set.of(TensorType.of(INT_64, 128))));
  }

  /**
   * Pins the <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a> {@code BiLSTM.call}
   * row on the vendored subject: the {@code BiLSTM} layer is vendored verbatim from {@code
   * kyzhouhzau/NLPGNN} ({@code nlpgnn/layers/bilstm.py}); only the driver is bespoke. The {@code
   * inputs} parameter (an integer token-ID tensor feeding a Keras {@code Embedding}) recovers
   * {@code (2, 5)} int32 exactly, flowing from the driver's {@code layer(tokens, training=False)}
   * call site through {@code tf.keras.layers.Layer.__call__} dispatch.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testBilstmCallVendored()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "bilstm_proj/nlpgnn/__init__.py",
          "bilstm_proj/nlpgnn/layers/__init__.py",
          "bilstm_proj/nlpgnn/layers/bilstm.py",
          "bilstm_proj/tf2_test_bilstm_call.py"
        },
        "nlpgnn/layers/bilstm.py",
        "BiLSTM.call",
        "bilstm_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Pins the <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a> {@code
   * TextCNN.predict} row on the vendored subject ({@code kyzhouhzau/NLPGNN}, {@code
   * nlpgnn/models/TextCNN.py}, same vendoring as {@link #testTextcnnCall()}): {@code predict}
   * forwards {@code inputs} to the model through {@code self(inputs, training)}. The {@code inputs}
   * parameter recovers {@code (2, 5)} int32 exactly; the second tracked local is the forward
   * result, float32 with ⊤ shape (the chained-layer body, wala/ML#358/wala/ML#530).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTextcnnPredict()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "textcnn_proj/nlpgnn/__init__.py",
          "textcnn_proj/nlpgnn/models/__init__.py",
          "textcnn_proj/nlpgnn/models/TextCNN.py",
          "textcnn_proj/tf2_test_textcnn_predict.py"
        },
        "nlpgnn/models/TextCNN.py",
        "TextCNN.predict",
        "textcnn_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Probe for <a href="https://github.com/wala/ML/issues/688">wala/ML#688</a>: a {@code map} stage
   * returning a tuple, batched with a tuple {@code padded_shapes}, iterated with destructuring —
   * the vendored gpt-2 {@code input_fn} element shape in miniature. Both halves type the
   * runtime-true {@code (4, 3) int64} (batch 4, padded to the longest sequence): the computed
   * second member resolves through the wala/ML#688 SSA fallback and the batch stage applies.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPaddedBatchPair()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_padded_batch_pair.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(DType.INT64, 4, 3))));
  }

  /**
   * Sibling half of {@link #testPaddedBatchPair()} (wala/ML#688): the second tuple member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPaddedBatchPair2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_padded_batch_pair.py",
        "consume2",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(DType.INT64, 4, 3))));
  }

  /**
   * Pins the <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a> {@code
   * TuckERLoader.target_convert} row on the vendored subject: the loader is vendored verbatim from
   * {@code kyzhouhzau/NLPGNN} ({@code nlpgnn/datas/graphloader.py}); the driver, the tiny {@code
   * data/} triple files, and the {@code nlpgnn/gnn/utils.py} reachable slice are bespoke. The
   * {@code targets} parameter (a {@code padded_batch} dict-element field) types {@code (2, ?)}
   * int32 — the declared {@code padded_shapes} dims under the batch dimension (<a
   * href="https://github.com/wala/ML/issues/673">wala/ML#673</a>) — unioned with the standard
   * partial-batch sibling {@code (?, ?)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTuckerTargetConvert()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "tucker_proj/nlpgnn/__init__.py",
          "tucker_proj/nlpgnn/gnn/__init__.py",
          "tucker_proj/nlpgnn/gnn/utils.py",
          "tucker_proj/nlpgnn/datas/__init__.py",
          "tucker_proj/nlpgnn/datas/graphloader.py",
          "tucker_proj/tf2_test_tucker_target_convert.py"
        },
        "nlpgnn/datas/graphloader.py",
        "TuckERLoader.target_convert",
        "tucker_proj",
        1,
        7,
        Map.of(
            3,
            Set.of(
                new TensorType(INT_32, asList(new NumericDim(2), DynamicDim.INSTANCE)),
                new TensorType(INT_32, asList(new SymbolicDim("?"), DynamicDim.INSTANCE)))));
  }

  /**
   * Whole-project-layout probe for <a href="https://github.com/wala/ML/issues/678">wala/ML#678</a>:
   * the subject's structure in miniature — nested entry scripts ({@code tests/TG/EN/}) each
   * defining a same-named {@code GenGPT2} over a root-level {@code nlpgnn} package (inner {@code
   * gpt2.GPT2} model, shared closure-dispatching {@code samples.sample_sequence}) — the layout the
   * fixture-scale reproductions lacked. Both siblings keep their call-graph nodes and type — the
   * layout is excluded as the wala/ML#678 trigger — with the wala/ML#685 cross-sibling closure
   * union as the pinned imprecision.
   *
   * <p>TODO: Expect each sibling's own shape once <a
   * href="https://github.com/wala/ML/issues/685">wala/ML#685</a> keys closure callees on the
   * dispatched function object.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnSliceGeneration()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "nlpgnn_slice_proj/nlpgnn/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/callbacks.py",
          "nlpgnn_slice_proj/nlpgnn/models/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/models/gpt2.py",
          "nlpgnn_slice_proj/nlpgnn/sample/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/sample/samples.py",
          "nlpgnn_slice_proj/tests/__init__.py",
          "nlpgnn_slice_proj/tests/TG/__init__.py",
          "nlpgnn_slice_proj/tests/TG/EN/__init__.py",
          "nlpgnn_slice_proj/tests/TG/EN/generation.py",
          "nlpgnn_slice_proj/tests/TG/EN/interactive.py"
        },
        "tests/TG/EN/generation.py",
        "GenGPT2.predict",
        "nlpgnn_slice_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_2_FLOAT32, TENSOR_3_3_FLOAT32)));
  }

  /**
   * Sibling half of {@link #testNlpgnnSliceGeneration()} (wala/ML#678) — the {@code interactive}
   * entry script, the one degraded in the whole-project consumer runs.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnSliceInteractive()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "nlpgnn_slice_proj/nlpgnn/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/callbacks.py",
          "nlpgnn_slice_proj/nlpgnn/models/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/models/gpt2.py",
          "nlpgnn_slice_proj/nlpgnn/sample/__init__.py",
          "nlpgnn_slice_proj/nlpgnn/sample/samples.py",
          "nlpgnn_slice_proj/tests/__init__.py",
          "nlpgnn_slice_proj/tests/TG/__init__.py",
          "nlpgnn_slice_proj/tests/TG/EN/__init__.py",
          "nlpgnn_slice_proj/tests/TG/EN/generation.py",
          "nlpgnn_slice_proj/tests/TG/EN/interactive.py"
        },
        "tests/TG/EN/interactive.py",
        "GenGPT2.predict",
        "nlpgnn_slice_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_2_FLOAT32, TENSOR_3_3_FLOAT32)));
  }

  /**
   * The complete NLPGNN subject vendored verbatim under {@code nlpgnn_full_proj} (wala/ML#690);
   * shared by {@link #testNlpgnnFullGeneration()} and {@link #testNlpgnnFullInteractive()} so the
   * two sibling guards cannot diverge as the fixture changes.
   */
  private static final String[] NLPGNN_FULL_PROJECT_FILES = {
    "nlpgnn_full_proj/nlpgnn/__init__.py",
    "nlpgnn_full_proj/nlpgnn/abandoned/GCNConvv0.py",
    "nlpgnn_full_proj/nlpgnn/abandoned/__init__.py",
    "nlpgnn_full_proj/nlpgnn/abandoned/scatter.py",
    "nlpgnn_full_proj/nlpgnn/bpemd/bpe.py",
    "nlpgnn_full_proj/nlpgnn/callbacks.py",
    "nlpgnn_full_proj/nlpgnn/datas/__init__.py",
    "nlpgnn_full_proj/nlpgnn/datas/checkpoint.py",
    "nlpgnn_full_proj/nlpgnn/datas/dataloader.py",
    "nlpgnn_full_proj/nlpgnn/datas/graphloader.py",
    "nlpgnn_full_proj/nlpgnn/datas/word2vec.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GAAEConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GATConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GCNConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GINConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/GSConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/RGCNConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/TGCNConv.py",
    "nlpgnn_full_proj/nlpgnn/gnn/__init__.py",
    "nlpgnn_full_proj/nlpgnn/gnn/glob.py",
    "nlpgnn_full_proj/nlpgnn/gnn/messagepassing.py",
    "nlpgnn_full_proj/nlpgnn/gnn/utils.py",
    "nlpgnn_full_proj/nlpgnn/layers/__init__.py",
    "nlpgnn_full_proj/nlpgnn/layers/albert_transformer.py",
    "nlpgnn_full_proj/nlpgnn/layers/attention.py",
    "nlpgnn_full_proj/nlpgnn/layers/bilstm.py",
    "nlpgnn_full_proj/nlpgnn/layers/decoder.py",
    "nlpgnn_full_proj/nlpgnn/layers/dense.py",
    "nlpgnn_full_proj/nlpgnn/layers/embedding.py",
    "nlpgnn_full_proj/nlpgnn/layers/gpt2_transformer.py",
    "nlpgnn_full_proj/nlpgnn/layers/normalization.py",
    "nlpgnn_full_proj/nlpgnn/layers/transformer.py",
    "nlpgnn_full_proj/nlpgnn/metrics/Losess.py",
    "nlpgnn_full_proj/nlpgnn/metrics/Metric.py",
    "nlpgnn_full_proj/nlpgnn/metrics/__init__.py",
    "nlpgnn_full_proj/nlpgnn/metrics/crf.py",
    "nlpgnn_full_proj/nlpgnn/metrics/type.py",
    "nlpgnn_full_proj/nlpgnn/models/GAAE.py",
    "nlpgnn_full_proj/nlpgnn/models/GAT.py",
    "nlpgnn_full_proj/nlpgnn/models/GCN.py",
    "nlpgnn_full_proj/nlpgnn/models/GIN.py",
    "nlpgnn_full_proj/nlpgnn/models/GraphSage.py",
    "nlpgnn_full_proj/nlpgnn/models/PCNN.py",
    "nlpgnn_full_proj/nlpgnn/models/RGCN.py",
    "nlpgnn_full_proj/nlpgnn/models/TextCNN.py",
    "nlpgnn_full_proj/nlpgnn/models/TextGCN2019.py",
    "nlpgnn_full_proj/nlpgnn/models/__init__.py",
    "nlpgnn_full_proj/nlpgnn/models/albert.py",
    "nlpgnn_full_proj/nlpgnn/models/bert.py",
    "nlpgnn_full_proj/nlpgnn/models/gpt2.py",
    "nlpgnn_full_proj/nlpgnn/models/tucker.py",
    "nlpgnn_full_proj/nlpgnn/optimizers/__init__.py",
    "nlpgnn_full_proj/nlpgnn/optimizers/optim.py",
    "nlpgnn_full_proj/nlpgnn/sample/__init__.py",
    "nlpgnn_full_proj/nlpgnn/sample/samples.py",
    "nlpgnn_full_proj/nlpgnn/savers.py",
    "nlpgnn_full_proj/nlpgnn/tokenizers/__init__.py",
    "nlpgnn_full_proj/nlpgnn/tokenizers/gpt2_tokenization.py",
    "nlpgnn_full_proj/nlpgnn/tokenizers/tokenization.py",
    "nlpgnn_full_proj/nlpgnn/tools.py",
    "nlpgnn_full_proj/setup.py",
    "nlpgnn_full_proj/tests/CLS/ALBERT/albert_cls_test.py",
    "nlpgnn_full_proj/tests/CLS/ALBERT/albert_cls_train.py",
    "nlpgnn_full_proj/tests/CLS/BERT/bert_classification_test.py",
    "nlpgnn_full_proj/tests/CLS/BERT/bert_classification_train.py",
    "nlpgnn_full_proj/tests/CLS/BilstmAttention/bilstm_attention_test.py",
    "nlpgnn_full_proj/tests/CLS/BilstmAttention/bilstm_attention_train.py",
    "nlpgnn_full_proj/tests/CLS/TextCNN/text_cnn_test.py",
    "nlpgnn_full_proj/tests/CLS/TextCNN/text_cnn_train.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/attention.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/bert.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/build_graph.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/train_text_gcn.py",
    "nlpgnn_full_proj/tests/GNN/BERT-TextGCN/transformer.py",
    "nlpgnn_full_proj/tests/GNN/auto_encoder/GAAE.py",
    "nlpgnn_full_proj/tests/GNN/gnn_for_nlp/text_gcn.py",
    "nlpgnn_full_proj/tests/GNN/gnn_for_nlp/text_sage.py",
    "nlpgnn_full_proj/tests/GNN/nodes_graph_classfication/train_gan.py",
    "nlpgnn_full_proj/tests/GNN/nodes_graph_classfication/train_gcn.py",
    "nlpgnn_full_proj/tests/GNN/nodes_graph_classfication/train_gin.py",
    "nlpgnn_full_proj/tests/GNN/nodes_graph_classfication/train_graphsage.py",
    "nlpgnn_full_proj/tests/KG2E/run_tucker.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/albert_ner_test.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/albert_ner_train.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/bert_ner_test.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/bert_ner_train.py",
    "nlpgnn_full_proj/tests/NER/NER_EN/data_processing.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/bert_ner_crf_test.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/bert_ner_crf_train.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/bert_ner_test.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/bert_ner_train.py",
    "nlpgnn_full_proj/tests/NER/NER_ZH/ner_data_preprocess.py",
    "nlpgnn_full_proj/tests/TG/EN/generation.py",
    "nlpgnn_full_proj/tests/TG/EN/interactive.py"
  };

  /**
   * The complete MusicTransformer-tensorflow2.0 subject vendored verbatim under {@code
   * musictransformer_proj} (wala/ML#683, wala/ML#684), so its whole-project guards analyze the
   * subject shape rather than a distilled fixture.
   */
  private static final String[] MUSICTRANSFORMER_PROJECT_FILES = {
    "musictransformer_proj/custom/callback.py",
    "musictransformer_proj/custom/layers.py",
    "musictransformer_proj/data.py",
    "musictransformer_proj/deprecated/seq_test.py",
    "musictransformer_proj/deprecated/sequence.py",
    "musictransformer_proj/deprecated/train.py",
    "musictransformer_proj/dist_train.py",
    "musictransformer_proj/generate.py",
    "musictransformer_proj/model.py",
    "musictransformer_proj/params.py",
    "musictransformer_proj/preprocess.py",
    "musictransformer_proj/train.py",
    "musictransformer_proj/utils.py"
  };

  /**
   * Verbatim whole-project guard for <a
   * href="https://github.com/wala/ML/issues/690">wala/ML#690</a>: the full NLPGNN subject vendored
   * as-is (all 94 {@code .py} files, matching the consumer's whole-project run; no added {@code
   * __init__.py}s under {@code tests/}). Before the fix, whichever same-named {@code GenGPT2}
   * sibling's closure reached the shared {@code sample_sequence.step} node second lost its lexical
   * {@code model} wiring (WALA's one-shot {@code visitLexical} snapshot), so its {@code
   * predict}/{@code call} never dispatched and its method nodes vanished at whole-project scale.
   * This pins the generation sibling's {@code predict} node and its parameter type, which must be
   * symmetric with {@link #testNlpgnnFullInteractive()}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullGeneration()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    runNlpgnnFullGeneration();
  }

  /**
   * Runs the NLPGNN whole-project generation analysis with its call-graph and type assertions.
   * Package-visible so {@link DiagnosticLoggingVolumeTest} can rerun the analysis under {@code
   * FINEST} without invoking another class's {@code @Test} method.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  void runNlpgnnFullGeneration()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "tests/TG/EN/generation.py",
        "GenGPT2.predict",
        "nlpgnn_full_proj",
        1,
        1,
        Map.of(3, Set.of(new TensorType(UNKNOWN, asList(DynamicDim.INSTANCE, new NumericDim(1))))));
  }

  /**
   * Sibling half of {@link #testNlpgnnFullGeneration()} (wala/ML#690) — the {@code interactive}
   * entry script, the one whose {@code predict}/{@code call} nodes vanished in the consumer's
   * whole-project run. Its {@code predict} parameter typing must be symmetric with the generation
   * sibling's.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNlpgnnFullInteractive()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        NLPGNN_FULL_PROJECT_FILES,
        "tests/TG/EN/interactive.py",
        "GenGPT2.predict",
        "nlpgnn_full_proj",
        1,
        1,
        Map.of(3, Set.of(new TensorType(UNKNOWN, asList(DynamicDim.INSTANCE, new NumericDim(1))))));
  }

  /**
   * Same-name-class guard for <a href="https://github.com/wala/ML/issues/678">wala/ML#678</a>: two
   * sibling scripts each define a Keras subclass named {@code GenGPT2} (with a {@code
   * super(GenGPT2, self)} by-name reference in {@code __init__}, mirroring the NLPGNN subject);
   * this pins that the first script's class keeps its call-graph nodes and its {@code predict}
   * parameter types.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSamenameA()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "samename_proj/tf2_test_samename_a.py", "samename_proj/tf2_test_samename_b.py"
        },
        "tf2_test_samename_a.py",
        "GenGPT2.predict",
        "samename_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Sibling half of {@link #testSamenameA()}: the second script's same-named class keeps its
   * call-graph nodes and typing too (wala/ML#678).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSamenameB()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "samename_proj/tf2_test_samename_a.py", "samename_proj/tf2_test_samename_b.py"
        },
        "tf2_test_samename_b.py",
        "GenGPT2.predict",
        "samename_proj",
        1,
        2,
        Map.of(3, Set.of(TENSOR_3_3_FLOAT32)));
  }

  /**
   * Deep variant of {@link #testSamenameA()} mirroring the wala/ML#678 subject's dispatch shape:
   * both sibling scripts pass their same-named model into one shared helper ({@code
   * helpers/samples.py}'s {@code sample_sequence}), whose nested closures capture {@code model}
   * lexically and dispatch {@code model.predict} from a frame reached by both scripts — the NLPGNN
   * {@code nlpgnn/sample/samples.py} structure the two-file fixture lacks. Dispatch survives the
   * whole chain (no wala/ML#678 node loss at this scale), but the closure bodies are
   * call-string-keyed, so one {@code step} node serves both lexical parents and each sink receives
   * the cross-sibling union rather than its own shape (runtime truth here: {@code (2, 2) float32}).
   *
   * <p>TODO: Expect exactly {@code (2, 2) float32} once <a
   * href="https://github.com/wala/ML/issues/685">wala/ML#685</a> keys closure callees on the
   * dispatched function object, the closure analogue of wala/ML#679.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSamenameDeepA()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "samename_deep_proj/helpers/__init__.py",
          "samename_deep_proj/helpers/samples.py",
          "samename_deep_proj/tf2_test_samename_deep_a.py",
          "samename_deep_proj/tf2_test_samename_deep_b.py"
        },
        "tf2_test_samename_deep_a.py",
        "consume",
        "samename_deep_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_2_FLOAT32, TENSOR_3_3_FLOAT32)));
  }

  /**
   * Sibling half of {@link #testSamenameDeepA()} (wala/ML#678): runtime truth is {@code (3, 3)
   * float32}; the extra member is the wala/ML#685 cross-sibling closure union.
   *
   * <p>TODO: Expect exactly {@code (3, 3) float32} once <a
   * href="https://github.com/wala/ML/issues/685">wala/ML#685</a> keys closure callees on the
   * dispatched function object.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSamenameDeepB()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "samename_deep_proj/helpers/__init__.py",
          "samename_deep_proj/helpers/samples.py",
          "samename_deep_proj/tf2_test_samename_deep_a.py",
          "samename_deep_proj/tf2_test_samename_deep_b.py"
        },
        "tf2_test_samename_deep_b.py",
        "consume",
        "samename_deep_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_2_FLOAT32, TENSOR_3_3_FLOAT32)));
  }

  /**
   * Pins the <a href="https://github.com/wala/ML/issues/676">wala/ML#676</a> subject: {@code
   * DynamicPositionEmbedding.call}'s {@code inputs} parameter on the vendored {@code
   * jason9693/MusicTransformer-tensorflow2.0} {@code custom/layers.py}. With {@code
   * tf.keras.layers.Embedding} modeled and constructor keyword arguments forwarded (wala/ML#664),
   * the chain composes concretely: tokens {@code (2, 50)} int32 through the embedding to {@code (2,
   * 50, 64)} float32 into the position encoding. On 0.52.13 the parameter was tensor- classified
   * but shapeless ({@code ? of unknown}, an integer-arithmetic seed leaking through the pointer
   * analysis); wala/ML#664 removed the leak and this modeling supplies the honest chain.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testMusicTransformerPositionEmbedding()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "musictx_proj/custom/__init__.py",
          "musictx_proj/custom/layers.py",
          "musictx_proj/tf2_test_musictx_encoder.py"
        },
        "custom/layers.py",
        "DynamicPositionEmbedding.call",
        "musictx_proj",
        1,
        6,
        Map.of(3, Set.of(TensorType.of(FLOAT_32, 2, 50, 64))));
  }

  @Test
  public void testModelAttributes()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes2.py",
        "f",
        1,
        1,
        Map.of(
            2, Set.of(TENSOR_3_4_FLOAT32, TENSOR_4_FLOAT32, TENSOR_4_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes3.py",
        "f",
        1,
        1,
        Map.of(
            2, Set.of(TENSOR_3_4_FLOAT32, TENSOR_4_FLOAT32, TENSOR_4_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes4.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes5.py",
        "f",
        1,
        1,
        Map.of(
            2, Set.of(TENSOR_3_4_FLOAT32, TENSOR_4_FLOAT32, TENSOR_4_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  @Test
  public void testModelAttributes6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes6.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  /**
   * Tests precise dataflow tracing for Keras {@code Model} weights when using keyword arguments for
   * both {@code Dense} layer instantiation and {@code Model} construction. Verifies that the
   * weights are correctly identified and have the expected shapes {@code (64, 5)} and {@code (5,)}.
   */
  @Test
  public void testModelAttributes7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes7.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  /**
   * Multi-Model precision regression: two distinct {@code tf.keras.models.Model(...)} calls in one
   * fixture, two sink functions, disjoint shapes per model. Validates that under the current
   * modeling each sink's parameter sees only its own model's weight shapes (not the union across
   * both models). See wala/ML#380's discussion of `Model.read_data` materialization. Companion to
   * {@link #testModelAttributesMultiModel2()} (same fixture, second sink). Disjoint dim choices
   * (64/5 vs 32/7) make a precision regression mechanically detectable: a "shapes unioned across
   * models" failure mode produces the 4-element set, not a 2-element subset.
   */
  @Test
  public void testModelAttributesMultiModel()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes_multi.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  /**
   * Companion to {@link #testModelAttributesMultiModel()} — pins the second sink's parameter to the
   * second model's weight shapes only.
   */
  @Test
  public void testModelAttributesMultiModel2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes_multi.py",
        "g",
        1,
        1,
        Map.of(2, Set.of(TENSOR_32_7_FLOAT32, TENSOR_7_FLOAT32)));
  }

  /**
   * Multi-Model separation with one extra call-chain frame: both Models are constructed inside a
   * {@code make_model(units)} helper, so both user-side calls of {@code make_model(...)} share the
   * same call site for {@code Model.do} (the one inside {@code make_model}). Call strings alone
   * collapsed both user models into one allocation context, unioning both models' weight shapes at
   * every sink; the receiver-keyed trampoline contexts of <a
   * href="https://github.com/wala/ML/issues/679">wala/ML#679</a> keep each model's dispatch chain
   * separate, so each sink now sees exactly its own model's weight shapes.
   */
  @Test
  public void testModelAttributesMultiModelWrapped()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes_multi_wrapped.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)));
  }

  /**
   * Companion to {@link #testModelAttributesMultiModelWrapped()} — same fixture, second sink,
   * pinned to the second model's weight shapes only (wala/ML#679).
   */
  @Test
  public void testModelAttributesMultiModelWrapped2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_attributes_multi_wrapped.py",
        "g",
        1,
        1,
        Map.of(2, Set.of(TENSOR_64_7_FLOAT32, TENSOR_7_FLOAT32)));
  }

  /**
   * Multi-Model separation on the {@code model(x)} call-output path: two distinct Models
   * constructed inside a {@code make_and_call(units, x)} helper that also performs the call. Each
   * user-side {@code make_and_call(...)} returns the model's output, with disjoint shapes (5 vs 7)
   * per call. Call strings alone collapsed both models into one context for {@code
   * Model.__call__}'s output allocation, unioning {@code {(20, 5), (20, 7)}} at every sink; the
   * receiver-keyed trampoline contexts of <a
   * href="https://github.com/wala/ML/issues/679">wala/ML#679</a> keep each model's dispatch chain
   * separate, so each sink now sees exactly its own model's output shape — same mechanism as {@link
   * #testModelAttributesMultiModelWrapped()}, on the call-output path instead of the
   * trainable-weights path.
   */
  @Test
  public void testModelCallMultiModelWrapped()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_model_call_multi_wrapped.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_20_5_FLOAT32)));
  }

  /**
   * Companion to {@link #testModelCallMultiModelWrapped()} — same fixture, second sink, pinned to
   * the second model's output shape only (wala/ML#679).
   */
  @Test
  public void testModelCallMultiModelWrapped2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_model_call_multi_wrapped.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_20_7_FLOAT32)));
  }

  /**
   * {@code replica_fn(input)} body: {@code return input * 2.0}. Both {@code input} (parameter) and
   * the binop result are tensors, so the expected count is 2 (1 param + 1 binop-result SSA value).
   * Prior to wala/ML#395's scalar-literal-broadcast fix, the binop result was under-classified
   * (null shape) and didn't register, producing a count of 1. The updated count reflects the
   * corrected identification.
   */
  @Test
  public void testCallbacks()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_callbacks.py", "replica_fn", 1, 2, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** See {@link #testCallbacks()} for the count rationale. */
  @Test
  public void testCallbacks2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_callbacks2.py", "replica_fn", 1, 2, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
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
   * Keyword-only parameters are modeled as formal parameters (<a
   * href="https://github.com/wala/ML/issues/596">wala/ML#596</a>). {@code f(x, *, y)} is called
   * {@code f(tf.constant(1), y=tf.ones([2, 3]))}; the keyword-only {@code y} must be a formal so
   * the call-site keyword argument binds to it. {@code consume(y)} pins {@code y}'s type, which is
   * therefore {@code (2, 3) float32}. Before the fix, {@code y} had no value number and {@code
   * consume} saw no tensor parameter.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testKwonlyParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_kwonly_param.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
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
   * Pins a slice-subscript result flowing into a dataset (wala/ML#400). Python {@code a_sliced =
   * a[:2, ..., tf.newaxis]; from_tensor_slices((a_sliced, y))} over a {@code (3, 2)} tensor: the
   * subscript is {@code (2, 2, 1)} (slice, ellipsis-fill, newaxis), so each iterated element is
   * {@code (2, 1)}. Because the {@code slice} builtin returns its receiver ({@code
   * Either.forRight(2)}), {@code a_sliced} aliases {@code a}; {@link
   * DatasetFromTensorSlicesGenerator} recovers the slice's shape by dispatching {@link
   * com.ibm.wala.cast.python.ml.client.SliceBuiltinOperation} on the stored value rather than
   * reading the receiver-aliased field PTS.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSliceSubscriptThroughDataset()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_iso_slicesub_ds.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_1_FLOAT32)));
  }

  /**
   * Regression guard for {@code tf.distribute.MirroredStrategy.distribute_datasets_from_function}'s
   * callback registration (wala/ML#113). The fixture registers {@code dataset_fn} only via {@code
   * strategy.distribute_datasets_from_function(dataset_fn)} &mdash; no other call site. The {@code
   * test(...)} helper's "function must exist in call graph" assertion fails pre-fix because {@code
   * dataset_fn} never gets traced. After this PR's `tensorflow.xml` fix (synthetic-method body on
   * {@code tensorflow/distribute/run/distribute_datasets_from_function} that allocates a stub
   * {@code InputContext} and invokes {@code dataset_fn(ctx)}), the callback enters the call graph
   * and the helper finds it. {@code input_context} is non-tensor, hence 0 tensor parameters; the 6
   * tensor variables in the body come from the chained {@code tf.data.Dataset} calls.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCallbacks3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_callbacks3.py", "dataset_fn", 0, 6, Map.of());
  }

  /**
   * {@code train_step(images, generator, discriminator, ...)} receives {@code image_batch} from the
   * training loop inside {@code train}. {@code image_batch} comes from iterating a dataset built
   * from mnist data via {@code train_images[..., None].astype(np.float32)} and {@code
   * from_tensor_slices(...).shuffle(...).batch(256)}. At runtime {@code image_batch} has shape in
   * {@code {(256, 28, 28, 1), (96, 28, 28, 1)}} dtype {@code float32} (60000 training images / 256
   * = 234 full batches + 1 partial batch of 96; verified by Python assert statements in {@code
   * tensorflow_gan_tutorial.py}).
   *
   * <p>Expected tensor variable count: 7. After wala/ML#430's {@code Gradient} generator allocates
   * a fresh tensor per {@code tape.gradient(...)} call instead of aliasing {@code sources}, each of
   * the two gradient calls in {@code train_step} (one for the generator, one for the discriminator)
   * registers an additional local tensor variable, lifting the count from the prior master baseline
   * of 5 to 7. Value 2 ({@code images}) is inferred concretely as {@code {(256, 28, 28, 1), (96,
   * 28, 28, 1)} float32}: the mnist pipeline resolves end to end. {@code mnist.load_data()}
   * supplies {@code (60000, 28, 28)} (<a
   * href="https://github.com/wala/ML/issues/361">wala/ML#361</a>), {@code [..., None]} and the
   * {@code (x - 127.5) / 127.5} binop chain carry it to {@code (60000, 28, 28, 1)}, {@code
   * from_tensor_slices} takes the element shape {@code (28, 28, 1)}, and {@code .batch(256)}
   * produces the two batch shapes (<a
   * href="https://github.com/wala/ML/issues/356">wala/ML#356</a>).
   */
  @Test
  public void testGanTutorial()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_gan_tutorial.py",
        "train_step",
        1,
        7,
        Map.of(2, Set.of(TENSOR_256_28_28_1_FLOAT32, TENSOR_96_28_28_1_FLOAT32)));
  }

  /**
   * Same structure as {@link #testGanTutorial()} but with {@code @tf.function} applied to {@code
   * train_step}. Runtime types for {@code image_batch} are identical and are verified via the
   * Python assert statements in {@code tensorflow_gan_tutorial.py} (not duplicated in {@code
   * tensorflow_gan_tutorial2.py} since the two files are structurally identical apart from the
   * decorator): shape in {@code {(256, 28, 28, 1), (96, 28, 28, 1)}}, dtype {@code float32}.
   * Expected count 7, same accounting as {@link #testGanTutorial()}: the two `tape.gradient(...)`
   * calls each contribute one fresh tensor variable post-wala/ML#430 (5 to 7). Value 2 is inferred
   * concretely as in {@link #testGanTutorial()}; the mnist binop-chain pipeline resolves (<a
   * href="https://github.com/wala/ML/issues/356">wala/ML#356</a>, <a
   * href="https://github.com/wala/ML/issues/361">wala/ML#361</a>).
   */
  @Test
  public void testGanTutorial2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_gan_tutorial2.py",
        "train_step",
        1,
        7,
        Map.of(2, Set.of(TENSOR_256_28_28_1_FLOAT32, TENSOR_96_28_28_1_FLOAT32)));
  }

  /**
   * Pins {@code generator_loss(fake_output)}'s parameter type. {@code fake_output} flows from
   * {@code discriminator(generated_images, ...)}; runtime shape is {@code (batch, 1) float32} since
   * the discriminator ends with a {@code Dense(1)} layer.
   *
   * <p>Inferred as {@code float32} with shape {@code ⊤} (unknown). The dtype is concrete and sound.
   * The shape is unknown rather than wrong: previously {@code ModelCall.getDefaultShapes} fell back
   * to the call's <em>input</em> shape when no output generator resolved, emitting the unsound
   * {@code (None, 100)} ({@code 100} is the generator's noise input dim from {@code
   * tf.keras.Input((100,))}, not the discriminator's {@code Dense(1)} output dim). A {@code
   * tf.keras.Model} generally transforms its input shape, so that fallback is removed
   * (wala/ML#537): the input shape now only refines a recovered output shape's batch dim, never
   * substitutes for it.
   *
   * <p>The runtime shape is {@code (256, 1)}/{@code (96, 1)} (verified by running the fixture:
   * {@code noise (256, 100)} -&gt; generator {@code (256, 28, 28, 1)} -&gt; discriminator {@code
   * (256, 1)}), confirming the old {@code (None, 100)} was the noise shape leaking through, not the
   * discriminator output.
   *
   * <p>TODO: tighten to {@code {(256, 1) float32, (96, 1) float32}}. The output node <em>is</em>
   * reachable now (the {@code outputs} construction argument points at the {@code Dense(1)} call),
   * but that {@code DenseCall} returns {@code null} shapes here because its input — the {@code
   * Flatten} of a {@code Conv2D} chain — isn't shape-tracked; recovering {@code (batch, 1)} needs
   * {@code DenseCall} output-shape inference through chained layer calls (tracked by <a
   * href="https://github.com/wala/ML/issues/358">wala/ML#358</a>), which in turn needs {@code
   * Conv2D}/{@code Flatten} output-shape modeling. Concrete batch dims 256/96 come from {@code
   * train_step}'s {@code images} parameter, per {@link #testGanTutorial}. (wala/ML#537 fixed the
   * unsound mis-propagation; this residual precision is wala/ML#358's domain.)
   */
  @Test
  public void testGanTutorialGeneratorLoss()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_gan_tutorial.py",
        "generator_loss",
        1,
        2,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Pins {@code top_p_logits(logits, p)}'s parameter type. Function body mirrors {@code
   * akanyaani/gpt-2-tensorflow2.0/sample.py}'s {@code top_p_logits}. Exercises several ops
   * currently routed through {@code ReadDataFallback} per wala/ML#449 ({@code tf.sort}, {@code
   * tf.cumsum}, {@code tf.stack}, {@code tf.range}, {@code tf.gather_nd}, {@code tf.where}), but
   * the parameter type of {@code logits} comes from its caller (a {@code tf.constant} with shape
   * {@code (1, 5)} dtype {@code float32}), so this test isolates caller-side propagation rather
   * than the body's op precision.
   *
   * <p>Empirically, {@code logits} is inferred as {@code (1, 5) float32} — concrete on both axes.
   * The caller's {@code tf.constant([[1.0, ..., 5.0]], dtype=tf.float32)} flows in cleanly, showing
   * that none of the body's {@code ReadDataFallback}-routed ops block caller-side propagation for
   * this function; the parameter type is fully resolved at the call site.
   */
  @Test
  public void testTopPLogits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_top_p_logits.py", "top_p_logits", 1, 11, Map.of(2, Set.of(TENSOR_1_5_FLOAT32)));
  }

  /**
   * Pins {@code _take_long_axis(arr, indices)}'s parameter types. Function body mirrors {@code
   * _take_long_axis} from {@code
   * LongmaoTeamTf/deep_recommenders/keras/models/retrieval/factorized_top_k.py}.
   *
   * <p>This fixture surfaced a real Ariadne bug: {@code tf.reshape(arr, tf.shape(other))} crashes
   * the analysis with {@code IllegalStateException} at {@code
   * TensorGenerator.getShapesFromShapeArgument} because the shape argument is a Tensor (the result
   * of {@code tf.shape(...)}) rather than a list/tuple literal. The {@code Reshape} generator
   * should gracefully degrade to ⊤ shape per the lattice conventions instead of throwing. Resolved
   * by <a href="https://github.com/wala/ML/issues/538">wala/ML#538</a>; parameter types pin
   * precisely.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/543">wala/ML#543</a>): the post-fix
   * local-tensor count of 9 captures every intermediate runtime-shape tensor flowing through the
   * body. Tighten once the body-level imprecision is addressed.
   */
  @Test
  public void testTakeAlongAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_take_along_axis.py",
        "_take_long_axis",
        2,
        9,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32), 3, Set.of(TENSOR_2_2_INT32)));
  }

  /**
   * Pins {@code multilayer_perceptron(x)}'s parameter type. Function body mirrors {@code
   * multilayer_perceptron} from {@code
   * YunYang1994/TensorFlow2.0-Examples/2-Basical_Models/Multilayer_Perceptron.py}. Uses raw {@code
   * tf.matmul} / {@code tf.add} / {@code tf.nn.sigmoid} / {@code tf.nn.softmax} against global
   * {@code tf.Variable} weights and biases — a different pattern from the {@code Dense}-layer
   * subclass-Model approach already covered by {@code testNeuralNetwork*}.
   *
   * <p>{@code x} is inferred as {@code (100, 784) float32}, flowing from the caller's {@code
   * batch_x = tf.constant(np.ones((100, 784), dtype=np.float32))}. This relies on the {@code numpy
   * → tf.constant} dtype/shape bridge fixed in <a
   * href="https://github.com/wala/ML/issues/539">wala/ML#539</a> (see {@link
   * #testConstantFromNumpy} for the isolated guard).
   *
   * <p>The local-tensor count is 16 (up from 14 before wala/ML#539): with {@code x} now typed, two
   * further {@code tf.matmul}/{@code tf.add} intermediates that consume it are recognized as
   * tensors.
   */
  @Test
  public void testMultilayerPerceptron()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_multilayer_perceptron.py",
        "multilayer_perceptron",
        1,
        16,
        Map.of(2, Set.of(TENSOR_100_784_FLOAT32)));
  }

  /**
   * Pins {@code logistic_regression(x)}'s parameter type. Function body mirrors {@code
   * logistic_regression} from {@code
   * aymericdamien/TensorFlow-Examples/.../2_BasicModels/logistic_regression.py}, a real-world
   * image-classification utility (logistic regression: {@code softmax(W x + b)} over global {@code
   * tf.Variable} weights and biases), for tensor-type inference coverage. Like {@link
   * #testMultilayerPerceptron}, it uses raw {@code tf.matmul} / {@code tf.nn.softmax} rather than
   * the {@code Dense}-layer subclass-{@code Model} approach of {@code testNeuralNetwork*}.
   *
   * <p>{@code x} is inferred as {@code (100, 784) float32}—both shape and dtype concrete—flowing
   * from the caller's {@code tf.constant(np.ones((100, 784), dtype=np.float32))} via the {@code
   * numpy→tf.constant} bridge (<a href="https://github.com/wala/ML/issues/539">wala/ML#539</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLogisticRegression()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_logistic_regression.py",
        "logistic_regression",
        1,
        6,
        Map.of(2, Set.of(TENSOR_100_784_FLOAT32)));
  }

  /**
   * Pins {@code nce_loss(x_embed, y)}'s parameter types. Function body mirrors {@code nce_loss}
   * from {@code aymericdamien/TensorFlow-Examples/.../2_BasicModels/word2vec.py}, a real-world
   * word-embedding utility (the averaged noise-contrastive-estimation loss over global {@code
   * tf.Variable} embedding/weight/bias matrices), for tensor-type inference coverage.
   *
   * <p>Both parameters are inferred concretely—shape and dtype: {@code x_embed} as {@code (4, 10)
   * float32} and {@code y} as {@code (4, 1) int32}, flowing from the {@code
   * tf.constant(np.ones(...))} call site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNceLoss()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nce_loss.py",
        "nce_loss",
        2,
        5,
        Map.of(2, Set.of(TENSOR_4_10_FLOAT32), 3, Set.of(TENSOR_4_1_INT32)));
  }

  /**
   * Pins {@code evaluate(x_embed)}'s parameter type. Function body mirrors {@code evaluate} from
   * {@code aymericdamien/TensorFlow-Examples/.../2_BasicModels/word2vec.py}, a real-world
   * word-embedding utility (the cosine similarity between an input embedding and every row of the
   * global embedding matrix), for tensor-type inference coverage.
   *
   * <p>{@code x_embed} is inferred as {@code (4, 10) float32}—both shape and dtype concrete—flowing
   * from the {@code tf.constant(np.ones(...))} call site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testEvaluate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_evaluate.py", "evaluate", 1, 13, Map.of(2, Set.of(TENSOR_4_10_FLOAT32)));
  }

  /**
   * Pins {@code random_jitter(input_image, real_image)}'s parameter types. Function body (and the
   * {@code resize}/{@code random_crop} helpers it calls) mirrors {@code random_jitter} from {@code
   * YunYang1994/TensorFlow2.0-Examples/.../Pix2Pix.py}, a real-world image-to-image translation
   * utility (random resize/crop/mirror data augmentation), for tensor-type inference coverage.
   *
   * <p>Both image parameters are inferred concretely—shape and dtype—as {@code (256, 256, 3)
   * float32}, flowing from the {@code tf.constant(np.ones(...))} call site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testRandomJitter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_random_jitter.py",
        "random_jitter",
        2,
        5,
        Map.of(2, Set.of(TENSOR_256_256_3_FLOAT32), 3, Set.of(TENSOR_256_256_3_FLOAT32)));
  }

  /**
   * Pins {@code MaskSparseCategoricalCrossentropy.__call__(y_true, y_predict, input_mask)}'s
   * parameter types. Class and method body mirror {@code MaskSparseCategoricalCrossentropy} from
   * {@code kyzhouhzau/NLPGNN/nlpgnn/metrics/Losess.py}, a real-world NLP utility (a mask-weighted
   * sparse-categorical-crossentropy loss), for tensor-type inference coverage. Unlike the {@code
   * tf.keras.Model.call} layer-chain methods ({@code testNeuralNetwork*}), this is a loss {@code
   * __call__} on a plain class that reduces its inputs to a scalar.
   *
   * <p>All three parameters are inferred concretely—shape and dtype: {@code y_true} as {@code (4,)
   * int32}, {@code y_predict} as {@code (4, 10) float32}, and {@code input_mask} as {@code (4,)
   * float32}, flowing from the {@code tf.constant(np.ones(...))} call site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testMaskedSparseCrossentropy()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_masked_sparse_ce.py",
        "MaskSparseCategoricalCrossentropy.__call__",
        3,
        6,
        Map.of(
            3, Set.of(TENSOR_4_INT32),
            4, Set.of(TENSOR_4_10_FLOAT32),
            5, Set.of(TENSOR_4_FLOAT32)));
  }

  /**
   * Pins {@code LSTM.call(x)}'s parameter type. Class and method body mirror the {@code LSTM}
   * recurrent model from {@code
   * aymericdamien/TensorFlow-Examples/.../3_NeuralNetworks/recurrent_network.py}, a real-world
   * sequence-classification utility (a built-in {@code tf.keras.layers.LSTM} followed by a {@code
   * Dense} read-out), for tensor-type inference coverage.
   *
   * <p>The input parameter {@code x} is recovered concretely on both axes—{@code (256, 28, 28)
   * float32}—flowing from the {@code lstm_net(x, is_training=True)} call site through {@code
   * tf.keras.Model.__call__} dispatch.
   *
   * <p>The forward-pass locals ({@code lstm_layer} output, {@code out} output, {@code softmax}) are
   * inferred as {@code float32} but with <em>unknown shape</em>: the built-in {@code LSTM}/{@code
   * Dense} output shapes are not narrowed (the layer-chain shape gap tracked by <a
   * href="https://github.com/wala/ML/issues/530">wala/ML#530</a>). The dtype axis—the load-bearing
   * one—is exact; only shape is ⊤.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testLstmCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_lstm_call.py", "LSTM.call", 1, 4, Map.of(3, Set.of(TENSOR_256_28_28_FLOAT32)));
  }

  /**
   * Pins {@code GCNLayer.call(node_embeddings, adjacency_lists)}'s tensor-parameter type. The
   * {@code GCNLayer} model and the {@code GraphConvolution}/{@code MessagePassing} layers it builds
   * on are vendored verbatim from {@code kyzhouhzau/NLPGNN} ({@code nlpgnn/models/GCN.py}, {@code
   * nlpgnn/gnn/GCNConv.py}, {@code nlpgnn/gnn/messagepassing.py}); only the driver and a
   * reachable-slice {@code nlpgnn/gnn/utils.py} (just the {@code GNNInput} named tuple) are
   * bespoke. This is a real-world graph-neural-network utility (a two-layer graph-convolution
   * message-passing model), exercised for tensor-type inference coverage across a multi-module
   * import chain.
   *
   * <p>The tensor parameter {@code node_embeddings} is recovered concretely on both axes—{@code (4,
   * 8) float32}—flowing from the driver's {@code model(node_embeddings, adjacency_lists,
   * training=False)} call site through {@code tf.keras.Model.__call__} dispatch, across the {@code
   * driver→GCN→GCNConv→MessagePassing} module boundaries. ({@code adjacency_lists} is a Python list
   * of edge tensors, not a tensor itself, so it is not a tensor parameter.)
   *
   * <p>The message-passing <em>output</em> locals (the {@code gc1}/{@code gc2} results) are still
   * inferred as ⊤, but the cause is now downstream of the matmul rather than the {@code NamedTuple}
   * field read. The aggregation ops are modeled, and the {@code GNNInput} {@code NamedTuple} field
   * read {@code node_embeddings = inputs.node_embeddings} recovers {@code (4, 8) float32} (read off
   * the instance field in the heap; <a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a>), so {@code tf.linalg.matmul} and
   * {@code propagate} inside {@code GraphConvolution.call} now type to {@code float32}. What
   * remains is the forward-result hop: {@code GraphConvolution.call} returns a typed tensor, but
   * that result does not propagate to the caller's {@code self.gc1(...)} local &mdash; the
   * user-subclass forward-result-typing gap (wala/ML#570, akin to <a
   * href="https://github.com/wala/ML/issues/595">wala/ML#595</a>). The decorated function's input
   * signature, the analysis goal, is nonetheless exact.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGcnCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gcn_proj/nlpgnn/__init__.py",
          "gcn_proj/nlpgnn/gnn/__init__.py",
          "gcn_proj/nlpgnn/gnn/utils.py",
          "gcn_proj/nlpgnn/gnn/messagepassing.py",
          "gcn_proj/nlpgnn/gnn/GCNConv.py",
          "gcn_proj/nlpgnn/models/__init__.py",
          "gcn_proj/nlpgnn/models/GCN.py",
          "gcn_proj/tf2_test_gcn_call.py"
        },
        "nlpgnn/models/GCN.py",
        "GCNLayer.call",
        "gcn_proj",
        1,
        6,
        Map.of(3, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins {@code GATLayer.call(node_embeddings, adjacency_lists)}'s tensor-parameter type. The
   * attention counterpart of {@link #testGcnCall()}: the {@code GATLayer} model and the {@code
   * GraphAttentionConvolution}/{@code MessagePassing} layers it builds on are vendored verbatim
   * from {@code kyzhouhzau/NLPGNN} ({@code nlpgnn/models/GAT.py}, {@code nlpgnn/gnn/GATConv.py},
   * {@code nlpgnn/gnn/messagepassing.py}); only the driver and a reachable-slice {@code
   * nlpgnn/gnn/utils.py} (the {@code GNNInput} named tuple plus {@code masksoftmax}/{@code
   * maybe_num_nodes}) are bespoke.
   *
   * <p>The tensor parameter {@code node_embeddings} is recovered concretely on both axes &mdash;
   * {@code (4, 8) float32} &mdash; flowing from the driver's {@code model(node_embeddings,
   * adjacency_lists, training=False)} call site through {@code tf.keras.Model.__call__} dispatch,
   * across the {@code driver→GAT→GATConv→MessagePassing} module boundaries. ({@code
   * adjacency_lists} is a Python list of edge tensors, not a tensor itself, so it is not a tensor
   * parameter.) As with {@link #testGcnCall()}, the decorated function's input signature &mdash;
   * the analysis goal &mdash; is exact, while the internal layer-output locals stay ⊤. The four
   * tracked tensor variables are {@code node_embeddings} (vn=3) and the {@code dropout1} output
   * (vn=11), both concrete {@code (4, 8) float32} (dropout preserves shape and dtype), plus the
   * {@code gc1}/{@code gc2} attention-convolution outputs (vn=18, vn=38), both ⊤ on each axis. The
   * layer outputs are ⊤ for the same reason as GCN: {@code GraphAttentionConvolution.call} unwraps
   * its input from a {@code GNNInput} {@code NamedTuple} field, which is not tracked as a tensor
   * (wala/ML#579), so the input arrives ⊤ and the attention aggregation through {@code
   * tf.math.unsorted_segment_*} (wala/ML#570, wala/ML#582) inherits it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGatCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gat_proj/nlpgnn/__init__.py",
          "gat_proj/nlpgnn/gnn/__init__.py",
          "gat_proj/nlpgnn/gnn/utils.py",
          "gat_proj/nlpgnn/gnn/messagepassing.py",
          "gat_proj/nlpgnn/gnn/GATConv.py",
          "gat_proj/nlpgnn/models/__init__.py",
          "gat_proj/nlpgnn/models/GAT.py",
          "gat_proj/tf2_test_gat_call.py"
        },
        "nlpgnn/models/GAT.py",
        "GATLayer.call",
        "gat_proj",
        1,
        4,
        Map.of(3, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Captured-gap reproduction for <a href="https://github.com/wala/ML/issues/659">wala/ML#659</a>:
   * in the vendored GAT subject, {@code maybe_num_nodes}'s {@code index} parameter (vn=2) is
   * currently <em>not</em> typed as a tensor, even though it receives a real tensor.
   *
   * <p>The chain: {@code MessagePassing._calculate_messages_all_type} computes {@code edge_targets
   * = adjanceny_list_edge_type[:, 1]} (a subscript-slice of an {@code enumerate(adjacency_lists)}
   * element), which flows through {@code GATConv.message_function}'s {@code edge_target} into
   * {@code masksoftmax(alpha, edge_target)} and then {@code maybe_num_nodes(index, ...)}. In {@code
   * propagate} the adjacency list types correctly to {@code (E, 2) int32}, but passed into {@code
   * _calculate_messages_all_type} the {@code adjacency_lists} parameter is context-collapsed on the
   * shared message-passing summary (merged with the {@code float32} {@code node_embeddings}),
   * losing its {@code int32}/rank-2 type. The corrupted ⊤ makes the {@code enumerate} element
   * empty-shaped, so the {@code [:, 1]} subscript returns ⊥ and {@code index} never types. This is
   * a regression from 0.52.9 (bisected to the {@code SliceBuiltinOperation} empty-shape change in
   * wala/ML#656, which unmasked the pre-existing collapse; ⊤ before, ⊥ after), and the root cause
   * is the same context-collapse class as <a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a>.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/659">wala/ML#659</a>): {@code index} should
   * type as a tensor once the shared-summary context collapse is resolved. Flip this to assert vn=2
   * <em>is</em> typed when it is fixed.
   */
  @Test
  public void testGatMaybeNumNodesIndexUntyped()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(
            getPathFiles("gat_proj"),
            new String[] {
              "gat_proj/nlpgnn/__init__.py",
              "gat_proj/nlpgnn/gnn/__init__.py",
              "gat_proj/nlpgnn/gnn/utils.py",
              "gat_proj/nlpgnn/gnn/messagepassing.py",
              "gat_proj/nlpgnn/gnn/GATConv.py",
              "gat_proj/nlpgnn/models/__init__.py",
              "gat_proj/nlpgnn/models/GAT.py",
              "gat_proj/tf2_test_gat_call.py"
            });
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    // Guard against a vacuous pass: the captured-gap assertion below only checks that vn=2 is
    // *absent* from the typed set, which is trivially satisfied if `maybe_num_nodes` is no longer
    // reached at all (e.g. the entrypoint or file list changes). Require a reachable node so the
    // test fails, rather than passing silently, when the reproduction stops exercising the target.
    assertTrue(
        "The reproduction must reach `maybe_num_nodes`; otherwise this captured-gap guard passes"
            + " vacuously (wala/ML#659).",
        CG.stream().anyMatch(n -> n.getMethod().getSignature().contains("maybe_num_nodes")));

    // Collect the parameter value numbers that `maybe_num_nodes` types as tensors. Its `index`
    // parameter is vn=2 (vn=1 is the function object).
    //
    // Coverage note: while the wala/ML#659 gap holds, no `maybe_num_nodes` parameter types, so the
    // `isParameter() && ...maybe_num_nodes` branch is never taken and `typedParamVns.add(...)` is
    // never reached (Codecov reports them as an uncovered line and partial branch). That is the
    // captured gap itself; both become covered when #659 is fixed and the assertion below is
    // flipped
    // to expect vn=2 typed.
    Set<Integer> typedParamVns = new HashSet<>();
    analysis.forEach(
        pt -> {
          if (pt.fst instanceof LocalPointerKey) {
            LocalPointerKey lpk = (LocalPointerKey) pt.fst;
            if (lpk.isParameter()
                && lpk.getNode().getMethod().getSignature().contains("maybe_num_nodes"))
              typedParamVns.add(lpk.getValueNumber());
          }
        });
    assertFalse(
        "Captured gap for wala/ML#659: `maybe_num_nodes`'s `index` (vn=2) is currently untyped"
            + " because `adjacency_lists` is context-collapsed on the shared message-passing"
            + " summary. Flip this assertion when the collapse is resolved.",
        typedParamVns.contains(2));
  }

  /**
   * Pins {@code GCN.call(self, features, adj)}'s tensor-parameter types. The {@code GCN} layer is
   * vendored verbatim from {@code LongmaoTeamTf/deep_recommenders} ({@code
   * deep_recommenders/keras/models/retrieval/gcn.py}); only the driver is bespoke. This is a
   * real-world graph-convolution layer, exercised for tensor-type inference coverage of a
   * <em>sparse tensor parameter</em>: unlike the other vendored layer methods, {@code adj} is a
   * {@code tf.SparseTensor}, on which the layer branches ({@code isinstance(adj, tf.SparseTensor)})
   * to use {@code tf.sparse.sparse_dense_matmul}. The driver feeds a sparse {@code adj}, mirroring
   * {@code train_gcn_on_cora_keras.py}'s {@code GCN(32)(feats, g)} call site where {@code g} is a
   * {@code scipy.sparse} adjacency.
   *
   * <p>Both parameters are recovered concretely: {@code features} (vn=3) as {@code (4, 8) float32}
   * and the sparse {@code adj} (vn=4) as {@code (4, 4) float32} with {@link
   * com.ibm.wala.cast.python.ml.types.TensorType.Layout#SPARSE} layout (a {@link
   * SparseTensorType}). So the sparse parameter is typed precisely, including its sparse layout,
   * rather than collapsing to dense or ⊤ &mdash; the sparse {@code TensorType} representation of <a
   * href="https://github.com/wala/ML/issues/588">wala/ML#588</a>. (The {@code **kwargs} parameter
   * is not a tensor and is not extracted.) Emitting {@code tf.SparseTensorSpec} for such a
   * parameter in an inferred signature is the downstream consumer's job; this confirms the typed
   * input it needs is available.
   *
   * <p>The local-tensor count rose from 4 to 7 when the Keras lazy-{@code build} protocol was
   * modeled (<a href="https://github.com/wala/ML/issues/595">wala/ML#595</a>): {@code self._kernel}
   * now dispatches, so {@code outputs} (the {@code Dense} result), the residual {@code outputs +=
   * features} value, and the return alias gained tensor types &mdash; a precision gain.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGcnSparseCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gcn_sparse_proj/deep_recommenders/__init__.py",
          "gcn_sparse_proj/deep_recommenders/keras/__init__.py",
          "gcn_sparse_proj/deep_recommenders/keras/models/__init__.py",
          "gcn_sparse_proj/deep_recommenders/keras/models/retrieval/__init__.py",
          "gcn_sparse_proj/deep_recommenders/keras/models/retrieval/gcn.py",
          "gcn_sparse_proj/tf2_test_gcn_sparse_call.py"
        },
        "deep_recommenders/keras/models/retrieval/gcn.py",
        "GCN.call",
        "gcn_sparse_proj",
        2,
        7,
        Map.of(3, Set.of(TENSOR_4_8_FLOAT32), 4, Set.of(SPARSE_TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the chained-layer forward result (<a href="https://github.com/wala/ML/issues/595">
   * wala/ML#595</a>): a value bound to a user-defined Keras {@code Layer}'s call is tensor-typed at
   * the call site, so it flows as a tensor into a downstream function. The fixture chains two
   * {@code GCN} layers (vendored from {@code deep_recommenders}), mirroring {@code
   * train_gcn_on_cora_keras.py}'s {@code x = GCN(32)(feats, g); GCN(num_classes)(x, g)}, and sinks
   * the first layer's output through {@code consume(hidden)}.
   *
   * <p>{@code hidden} (the {@code GCN.call} return, a {@code Dense} output) is tensor-classified
   * because the modeled Keras lazy-{@code build} protocol invokes {@code GCN.build}, giving the
   * {@code build()}-created {@code self._kernel} sublayer a points-to set. With constructor keyword
   * arguments forwarded to {@code __init__} (wala/ML#664) and the layer-method trampolines keyed on
   * the receiver instance (wala/ML#679), each instance's {@code build} constructs its own {@code
   * Dense}, so the runtime-true {@code (4, 16) float32} is inferred without the other instance's
   * spurious {@code (4, 8)}. The remaining {@code ? of float32} member comes from the statically
   * dead {@code outputs += features} residual branch ({@code self._residual} is constantly {@code
   * False}), which the path-insensitive analysis still evaluates.
   *
   * <p>TODO: Expect exactly {@code (4, 16) float32} once <a
   * href="https://github.com/wala/ML/issues/681">wala/ML#681</a> prunes branches guarded by
   * statically-constant instance fields.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGcnChainConsume()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gcn_chain_proj/deep_recommenders/__init__.py",
          "gcn_chain_proj/deep_recommenders/keras/__init__.py",
          "gcn_chain_proj/deep_recommenders/keras/models/__init__.py",
          "gcn_chain_proj/deep_recommenders/keras/models/retrieval/__init__.py",
          "gcn_chain_proj/deep_recommenders/keras/models/retrieval/gcn.py",
          "gcn_chain_proj/tf2_test_gcn_chain_call.py"
        },
        "tf2_test_gcn_chain_call.py",
        "consume",
        "gcn_chain_proj",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 16), TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Pins {@code TextCNN.call(inputs, training)}'s tensor-parameter type. The {@code TextCNN} model
   * is vendored verbatim from {@code kyzhouhzau/NLPGNN} ({@code nlpgnn/models/TextCNN.py}); only
   * the driver is bespoke. This is a real-world text-classification utility (a convolutional
   * sentence encoder: embedding lookup, parallel {@code Conv1D} kernels, global average pooling,
   * concatenation, batch normalization, and a softmax dense head), exercised for tensor-type
   * inference coverage. Unlike the GNN cohort ({@link #testGcnCall()}, {@link #testGatCall()}) and
   * the float-feature method archetypes, the decorated parameter {@code inputs} is an integer
   * token-ID tensor (an embedding-lookup index), so this measures int-dtype parameter recovery.
   *
   * <p>The tensor parameter {@code inputs} is recovered concretely on both axes &mdash; {@code (2,
   * 5) int32} &mdash; flowing from the driver's {@code model(inputs, training=False)} call site
   * through {@code tf.keras.Model.__call__} dispatch into {@code TextCNN.call}. As with the GNN
   * cohort, the decorated function's input signature &mdash; the analysis goal &mdash; is exact;
   * here it confirms that recovery holds for an integer-dtype parameter and a convolutional
   * (non-message-passing) body, not only the float-feature archetypes.
   *
   * <p>The tracked tensor variables are {@code inputs} (concrete {@code (2, 5) int32}), the
   * embedding output (concrete {@code (2, 5, 8)} float32 with {@code Embedding} modeled,
   * wala/ML#676), the convolution intermediate (⊤ on both axes &mdash; {@code Conv1D}/{@code
   * GlobalAvgPool1D} remain unmodeled), and the softmax {@code Dense} head's output (vn=72, {@code
   * float32} dtype but ⊤ shape &mdash; {@code DenseCall} hard-codes {@code float32} but loses the
   * shape through the chained-layer body, <a
   * href="https://github.com/wala/ML/issues/358">wala/ML#358</a>/<a
   * href="https://github.com/wala/ML/issues/530">wala/ML#530</a>). These residual-⊤ body locals are
   * pre-existing shape gaps, not new findings, and are downstream of the (exact) input signature.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTextcnnCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "textcnn_proj/nlpgnn/__init__.py",
          "textcnn_proj/nlpgnn/models/__init__.py",
          "textcnn_proj/nlpgnn/models/TextCNN.py",
          "textcnn_proj/tf2_test_textcnn_call.py"
        },
        "nlpgnn/models/TextCNN.py",
        "TextCNN.call",
        "textcnn_proj",
        1,
        4,
        Map.of(3, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Pins the output type of {@code tf.math.unsorted_segment_sum} (wala/ML#570). The output dtype
   * inherits from the {@code data} input ({@code float32}); the shape is the concrete {@code (2,
   * 3)} — {@code [num_segments] ++ data.shape[segment_ids.ndim:]} with the static {@code
   * num_segments = 2}, rank-1 {@code segment_ids}, and {@code (3, 3)} {@code data} (wala/ML#582).
   * Verified via a {@code consume_sum} sink on the aggregation result.
   */
  @Test
  public void testUnsortedSegmentSum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_unsorted_segment.py", "consume_sum", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins the output type of {@code tf.math.unsorted_segment_max} (wala/ML#570). Same dtype-from-
   * {@code data} and static-{@code num_segments} {@code (2, 3)} shape recovery as {@link
   * #testUnsortedSegmentSum} (wala/ML#582).
   */
  @Test
  public void testUnsortedSegmentMax()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_unsorted_segment.py", "consume_max", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins the output type of {@code tf.math.unsorted_segment_mean} (wala/ML#570). Same dtype-from-
   * {@code data} and static-{@code num_segments} {@code (2, 3)} shape recovery as {@link
   * #testUnsortedSegmentSum} (wala/ML#582).
   */
  @Test
  public void testUnsortedSegmentMean()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_unsorted_segment.py",
        "consume_mean",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins {@code Transformer.call(encoder_inputs, decoder_inputs)}'s parameter types. The {@code
   * Transformer} layer and the {@code MultiHeadAttention}/{@code ScaledDotProductAttention} layers
   * it builds on are vendored verbatim from {@code LongmaoTeamTf/deep_recommenders} ({@code
   * deep_recommenders/keras/models/nlp/transformer.py} and {@code .../multi_head_attention.py});
   * only the driver is bespoke. This is a real-world sequence-to-sequence utility (a
   * multi-head-attention encoder/decoder transformer), exercised for tensor-type inference coverage
   * across a multi-module import chain.
   *
   * <p>Both token-sequence parameters are recovered concretely on both axes—{@code (2, 5) int32}
   * each—flowing from the driver's {@code transformer(encoder_inputs, decoder_inputs)} call site
   * through {@code tf.keras.layers.Layer.__call__} dispatch, across the {@code
   * deep_recommenders→nlp→transformer} package boundaries.
   *
   * <p>With {@code tf.keras.backend} modeled (<a
   * href="https://github.com/wala/ML/issues/666">wala/ML#666</a>), the padding mask {@code masks =
   * K.equal(inputs, 0)} is a third function-local tensor, typed {@code (2, 5)} bool. With {@code
   * add_weight} consuming its arguments (wala/ML#667), {@code embeddings =
   * K.gather(self.embeddings, inputs)} is a fourth: the embedding table types {@code (?, 8)}
   * float32 and the {@code gather} pass-through carries it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTransformerCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "tr_proj/deep_recommenders/__init__.py",
          "tr_proj/deep_recommenders/keras/__init__.py",
          "tr_proj/deep_recommenders/keras/models/__init__.py",
          "tr_proj/deep_recommenders/keras/models/nlp/__init__.py",
          "tr_proj/deep_recommenders/keras/models/nlp/multi_head_attention.py",
          "tr_proj/deep_recommenders/keras/models/nlp/transformer.py",
          "tr_proj/tf2_test_transformer_call.py"
        },
        "deep_recommenders/keras/models/nlp/transformer.py",
        "Transformer.call",
        "tr_proj",
        2,
        4,
        Map.of(3, Set.of(TENSOR_2_5_INT32), 4, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Pins {@code crf_unary_score(tag_indices, sequence_lengths, inputs)}'s parameter types. Function
   * body mirrors {@code crf_unary_score} from {@code kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py}, a
   * real-world linear-chain CRF function exercised for tensor-type inference coverage. Its {@code
   * tf.reshape} with runtime-derived dimensions ({@code tf.shape(inputs)[0]}) previously crashed
   * the analysis (wala/ML#567).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfUnaryScore()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_unary_score",
        3,
        18,
        Map.of(
            2, Set.of(TENSOR_2_3_INT32),
            3, Set.of(TENSOR_2_INT32),
            4, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * Pins {@code crf_binary_score(tag_indices, sequence_lengths, transition_params)}'s parameter
   * types. Function body mirrors {@code crf_binary_score} from {@code
   * kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py} for tensor-type inference coverage.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfBinaryScore()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_binary_score",
        3,
        14,
        Map.of(
            2, Set.of(TENSOR_2_3_INT32),
            3, Set.of(TENSOR_2_INT32),
            4, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins {@code crf_sequence_score(inputs, tag_indices, sequence_lengths, transition_params)}'s
   * parameter types. Function body mirrors {@code crf_sequence_score} from {@code
   * kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py} for tensor-type inference coverage.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfSequenceScore()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_sequence_score",
        4,
        9,
        Map.of(
            2, Set.of(TENSOR_2_3_4_FLOAT32),
            3, Set.of(TENSOR_2_3_INT32),
            4, Set.of(TENSOR_2_INT32),
            5, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins {@code crf_log_norm(inputs, sequence_lengths, transition_params)}'s parameter types.
   * Function body mirrors {@code crf_log_norm} from {@code kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py}
   * for tensor-type inference coverage.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfLogNorm()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_log_norm",
        3,
        9,
        Map.of(
            2, Set.of(TENSOR_2_3_4_FLOAT32),
            3, Set.of(TENSOR_2_INT32),
            4, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the {@code tf.slice} output shape derived from constant {@code begin}/{@code size}
   * arguments (wala/ML#569). {@code tf.slice(x, [0, 1], [2, 2])} over a {@code (3, 4)} input yields
   * {@code (2, 2)} — all {@code size} entries are non-negative, so the output shape is {@code size}
   * exactly, independent of the input shape.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSlice()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_slice.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Pins the {@code tf.slice} output shape for the "all remaining" case (wala/ML#569). A {@code
   * size[i]} of {@code -1} resolves to {@code input.shape[i] - begin[i]}: {@code tf.slice(x, [1,
   * 0], [-1, 3])} over a {@code (3, 4)} input yields {@code (3 - 1, 3) = (2, 3)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSliceRemaining()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_slice.py", "consume_remaining", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins {@code crf_forward(inputs, state, transition_params, sequence_lengths)}'s parameter types.
   * Function body mirrors {@code crf_forward} from {@code kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py}
   * for tensor-type inference coverage. {@code crf_forward} is reached only through {@code
   * crf_log_norm} (its sole NLPGNN caller), which passes {@code inputs} and {@code state} from
   * {@code tf.slice}/{@code tf.squeeze} results over a {@code (2, 3, 4)} constant. {@code inputs}
   * (from {@code tf.slice(inputs, [0, 1, 0], [-1, -1, -1])}) infers as {@code (2, 2, 4)} via the
   * {@code begin}/{@code size} shape derivation (wala/ML#569). {@code state} (from {@code
   * tf.squeeze(tf.slice(inputs, [0, 0, 0], [-1, 1, -1]), [1])}) infers as {@code (2, 4)}: the
   * {@code Slice} shape gives {@code (2, 1, 4)} and {@code tf.squeeze}'s axis-1 removal
   * (wala/ML#513) drops the singleton. Both were previously {@code ⊤}-shaped (dtype only,
   * wala/ML#568).
   *
   * <p>The local tensor-variable count is {@code 15}: the two {@code tf.transpose} calls now
   * allocate distinct tensors rather than aliasing their inputs as first-argument {@code
   * pass_through} did (wala/ML#513 bucket 2a), so one additional reassignment is counted as a
   * tensor local.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfForward()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_forward",
        4,
        15,
        Map.of(
            2, Set.of(TENSOR_2_2_4_FLOAT32),
            3, Set.of(TENSOR_2_4_FLOAT32),
            4, Set.of(TENSOR_4_4_FLOAT32),
            5, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Pins the output shape of {@code tf.squeeze} with a named axis (wala/ML#513). {@code
   * tf.squeeze(x, [1])} over a {@code (2, 1, 3, 1)} tensor drops only axis 1: {@code (2, 3, 1)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSqueezeAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_squeeze.py", "consume_axis", 1, 1, Map.of(2, Set.of(TENSOR_2_3_1_FLOAT32)));
  }

  /**
   * Pins the output shape of {@code tf.squeeze} with no axis (wala/ML#513). {@code tf.squeeze(x)}
   * over a {@code (2, 1, 3, 1)} tensor drops every statically size-1 axis: {@code (2, 3)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSqueezeAll()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_squeeze.py", "consume_all", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins {@code tf.squeeze} with a single (non-list) integer axis (wala/ML#513). {@code
   * tf.squeeze(x, 1)} over a {@code (2, 1, 3, 1)} tensor drops axis 1: {@code (2, 3, 1)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSqueezeSingleAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_squeeze.py", "consume_single", 1, 1, Map.of(2, Set.of(TENSOR_2_3_1_FLOAT32)));
  }

  /**
   * Pins {@code tf.squeeze} with multiple named axes (wala/ML#513). {@code tf.squeeze(x, [1, 3])}
   * over a {@code (2, 1, 3, 1)} tensor drops both size-1 axes: {@code (2, 3)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSqueezeMultiAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_squeeze.py", "consume_multi", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins {@code crf_decode_forward(inputs, state, transition_params, sequence_lengths)}'s parameter
   * types. Function body mirrors {@code crf_decode_forward} from {@code
   * kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py} for tensor-type inference coverage. The caller passes
   * {@code inputs} from {@code x[:, 1:, :]} and {@code state} from {@code x[:, 0, :]}, recovered as
   * {@code (2, 2, 4)} and {@code (2, 4)} respectively via the multi-dim subscript-slice modeling
   * (wala/ML#406): {@code inputs[:, 1:, :]} drops the leading element of the middle axis ({@code 3
   * → 2}) and {@code inputs[:, 0, :]} drops the middle axis entirely (an integer index).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCrfDecodeForward()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_crf.py",
        "crf_decode_forward",
        4,
        6,
        Map.of(
            2, Set.of(TENSOR_2_2_4_FLOAT32),
            3, Set.of(TENSOR_2_4_FLOAT32),
            4, Set.of(TENSOR_4_4_FLOAT32),
            5, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Pins the output shape of a non-zero-start slice on the first axis of a multi-dim subscript
   * (wala/ML#406). {@code x[1:3, :, :]} over a {@code (4, 5, 6)} tensor keeps 2 rows and the
   * trailing axes intact: {@code (2, 5, 6)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSubscriptMultidimRows()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_subscript_multidim.py",
        "consume_rows",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_5_6_FLOAT32)));
  }

  /**
   * Pins the output shape of a middle-axis slice in a multi-dim subscript (wala/ML#406). {@code
   * x[:, 1:, :]} over a {@code (4, 5, 6)} tensor drops the leading element of the middle axis:
   * {@code (4, 4, 6)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSubscriptMultidimCols()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_subscript_multidim.py",
        "consume_cols",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_6_FLOAT32)));
  }

  /**
   * Pins the output shape of an integer index on the middle axis of a multi-dim subscript
   * (wala/ML#406). {@code x[:, 0, :]} over a {@code (4, 5, 6)} tensor drops the middle axis
   * entirely: {@code (4, 6)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSubscriptMultidimIndex()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_subscript_multidim.py",
        "consume_index",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_6_FLOAT32)));
  }

  /**
   * Pins the output shape of a multi-dim subscript that mixes a slice, an ellipsis, and a newaxis
   * (wala/ML#406). {@code a[:2, ..., tf.newaxis]} over a {@code (3, 2)} tensor slices the first
   * axis to 2, lets the ellipsis fill the remaining axis (2), and appends a size-1 axis for the
   * newaxis: {@code (2, 2, 1)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSubscriptNewaxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_subscript_newaxis.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_2_1_FLOAT32)));
  }

  /**
   * Pins {@code _gather_elements_along_row(data, column_indices)}'s parameter types. Function body
   * mirrors {@code _gather_elements_along_row} from {@code
   * deep_recommenders/keras/models/retrieval/sbcnm.py} (identical to {@code _take_long_axis} in
   * {@code factorized_top_k} per the source), a real-world recommender-systems utility, for
   * tensor-type inference coverage. Both parameters infer concretely: {@code data} as {@code (2, 4)
   * float32} and {@code column_indices} as {@code (2, 3) int32}. (The function's
   * runtime-dimensioned final {@code tf.reshape} leaves the local <em>result</em> symbolic, but the
   * parameters themselves are exact.)
   *
   * <p>The local tensor-variable count is {@code 9}: the {@code tf.tile} call now allocates a
   * distinct tensor rather than aliasing its input as first-argument {@code pass_through} did
   * (wala/ML#513 bucket 2a), so {@code row_indices} and a downstream use are additionally counted
   * as tensor locals.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGatherElementsAlongRow()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gather_elements_along_row.py",
        "_gather_elements_along_row",
        2,
        9,
        Map.of(
            2, Set.of(TENSOR_2_4_FLOAT32),
            3, Set.of(TENSOR_2_3_INT32)));
  }

  /**
   * Pins {@code create_attention_mask_from_input_mask(from_tensor, to_mask)}'s parameter types.
   * Function body mirrors {@code create_attention_mask_from_input_mask} from {@code
   * kyzhouhzau/NLPGNN/nlpgnn/tools.py}, a real-world function that builds a 3D attention mask from
   * a 2D input mask, for tensor-type inference coverage. Both parameters infer concretely.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCreateAttentionMaskFromInputMask()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_create_attention_mask.py",
        "create_attention_mask_from_input_mask",
        2,
        6,
        Map.of(
            2, Set.of(TENSOR_2_3_4_FLOAT32),
            3, Set.of(TENSOR_2_5_INT32)));
  }

  /**
   * Pins {@code accuracy(y_pred, y_true)}'s parameter types. Function body mirrors {@code accuracy}
   * from {@code YunYang1994/TensorFlow2.0-Examples/2-Basical_Models/Multilayer_Perceptron.py}.
   * Distinct from {@link #testNeuralNetwork4}'s {@code accuracy} (which is the {@code
   * Dense}-layer-chain variant from a different repo); this is the raw-{@code tf.matmul} MLP
   * companion, paired with {@link #testMultilayerPerceptron}.
   *
   * <p>Empirically, both parameters are concrete: {@code y_pred} (vn=2) is {@code (2, 2) float32}
   * and {@code y_true} (vn=3) is {@code (2,) int64}, matching the caller-side {@code tf.constant}
   * shapes.
   */
  @Test
  public void testMlpAccuracy()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_mlp_accuracy.py",
        "accuracy",
        2,
        7,
        Map.of(2, Set.of(TENSOR_2_2_FLOAT32), 3, Set.of(TENSOR_2_INT64)));
  }

  /**
   * Isolating regression guard for the {@code numpy → tf.constant} dtype/shape bridge (<a
   * href="https://github.com/wala/ML/issues/539">wala/ML#539</a>), surfaced from {@link
   * #testMultilayerPerceptron}. {@code consume(x)} where {@code x = tf.constant(np.ones((2, 3),
   * dtype=np.float32))} infers {@code (2, 3) float32}: {@code np.ones} is now modeled (see {@link
   * NpOnes}), so {@link Constant} recovers the numpy producer's shape and dtype through the value
   * argument rather than falling back to {@code ⊤ shape / UNKNOWN dtype}.
   */
  @Test
  public void testConstantFromNumpy()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_constant_from_numpy.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testConstantFromNumpy} for the {@code np.zeros → tf.constant} bridge:
   * {@code consume(x)} where {@code x = tf.constant(np.zeros((2, 3), dtype=np.int32))} infers
   * {@code (2, 3) int32}. Exercises {@link NpZeros} via the {@code createManualGenerator} recovery
   * path (symmetric to the {@code np.ones} bridge in {@link #testMultilayerPerceptron}, but with a
   * non-float dtype). Positive regression guard for wala/ML#539.
   */
  @Test
  public void testConstantFromNpZeros()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_constant_from_np_zeros.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
  }

  /**
   * {@code MyModel.call(self, x)} receives {@code x} from {@code model(images)} calls inside {@code
   * train_step} and {@code test_step}. {@code images} comes from iterating {@code train_ds}, {@code
   * valid_ds}, or {@code test_ds} &mdash; all created from mnist data via {@code
   * .astype(np.float32) / 255.0} and {@code [..., tf.newaxis]}, then batched by 32. At runtime
   * {@code x} has shape {@code (32, 28, 28, 1)} dtype {@code float32} (verified by Python assert
   * statements in {@code tensorflow_eager_execution.py}).
   *
   * <p>Note: {@code test_ds} yields mostly {@code (32, 28, 28, 1)} batches plus one trailing
   * partial batch of shape {@code (16, 28, 28, 1)} (since {@code 10000 % 32 == 16}), so the
   * aspirational union for value 3 includes both shapes.
   *
   * <p>The rule-based count is 5 (1 parameter + 4 intermediate layer-call ops {@code conv1}, {@code
   * flatten}, {@code d1}, {@code d2}); after the fix for wala/ML#358 (chained {@code Dense} shape
   * propagation), {@code d1} and {@code d2} are now tracked through the SSA-chain fallback. Counts
   * are source-level &mdash; one per distinct value number, deduplicated across the two calling
   * contexts ({@code train_step}/{@code test_step}; wala/ML#371, Option 2) &mdash; so the count is
   * 4 (parameter plus three registered intermediates) rather than the context-multiplied 8. The
   * residual gap from 4 to the rule-based 5 is one intermediate that still doesn't register; see
   * wala/ML#389.
   *
   * <p>With the count check passing, the test now fails on value 3's type: actual {@code {(32, 28)
   * float32, (16, 28) float32, (28, 28) float32, ? unknown}} &mdash; a union that contains an
   * over-peeled shape ({@code (32, 28)} / {@code (16, 28)} = batch applied to a peeled {@code
   * (28,)}), the unbatched source shape {@code (28, 28)}, and a ⊤ entry. The {@code float32} dtype
   * on three of the four entries indicates that per-index dtype dispatch routes {@code x_train}'s
   * dtype to slot 0 correctly; what is missing is (a) the shape contribution from the {@code
   * x_train[..., tf.newaxis]} subscript that would add the trailing {@code 1} dim, and (b)
   * suppression of the erroneous over-peel path. The earlier labels-swap symptom (shape {@code
   * (32,)} uint8) described under wala/ML#396 appears to have been resolved by the per-index dtype
   * delegation work landed on this branch; the remaining shape mismatch is shape-only.
   */
  @Test
  public void testEagerExecution()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tensorflow_eager_execution.py",
        "MyModel.call",
        1,
        4,
        Map.of(3, Set.of(TENSOR_32_28_28_1_FLOAT32, TENSOR_16_28_28_1_FLOAT32)));
  }

  /**
   * Parameter {@code x} of {@code NeuralNet.call} receives tensors from <b>four</b> source-level
   * call sites of {@code neural_net}, with three distinct runtime shapes (all {@code float32},
   * verified by Python {@code assert} statements in {@code neural_network.py}):
   *
   * <ul>
   *   <li>line 136 ({@code neural_net(x, ...)} inside {@code run_optimization}) &mdash; {@code
   *       (256, 784)} via {@code batch_x} forwarded through {@code x};
   *   <li>line 167 ({@code neural_net(batch_x, ...)} in the training loop) &mdash; {@code (256,
   *       784)};
   *   <li>line 189 ({@code neural_net(x_test, ...)}) &mdash; {@code (10000, 784)};
   *   <li>line 207 ({@code neural_net(test_images)} in the visualization block) &mdash; {@code (5,
   *       784)} via {@code x_test[:n_images]}.
   * </ul>
   *
   * <p>The aspirational expected set for value 3 is therefore the union {@code {(256, 784), (10000,
   * 784), (5, 784) float32}}. A downstream {@code @tf.function(input_signature=...)} consumer would
   * merge these into a single {@code tf.TensorSpec(shape=(None, 784), dtype=tf.float32)} using a
   * wildcard for the varying first dimension, so the union &mdash; not any individual shape &mdash;
   * is the correct source-level specification.
   *
   * <p>Value 3 is currently partially resolved. Shape inference through {@code np.array(x_train,
   * np.float32).reshape([-1, 784]) / 255.0} recovers a partial {@code (?, 784)} (the {@code -1}
   * slot stays symbolic when the receiver's shape is implicit-PK), and the batched shape {@code
   * (256, 784)} follows from {@code from_tensor_slices}'s per-index slice + {@code .batch(256)}.
   * Under the now-union-across-contexts helper, the aggregated actual for vn=3 is {@code {(?, 784)
   * float32, (256, 784) float32}} &mdash; one element per flow family, roughly one per trampoline
   * context. The test fails on types because the concrete test-set shape {@code (10000, 784)} and
   * the visualization slice shape {@code (5, 784)} never fall out of the reshape-{@code -1} slot:
   * the {@code (?, 784)} actual is a coarse approximation that covers both call sites. The
   * remaining analyzer gaps are (a) resolve {@code -1} in {@code reshape([-1, 784])} against the
   * source's concrete mnist-test shape {@code (10000, 28, 28)} to yield {@code (10000, 784)}, and
   * (b) propagate the constant-step slice {@code x_test[:n_images]} so {@code (5, 784)} falls out
   * of {@code (10000, 784)}.
   *
   * <p>Rule-based tensor variable count is 5 (1 parameter {@code x} + 4 intermediate ops {@code
   * fc1}, {@code fc2}, {@code out}, {@code softmax}). With the fix for wala/ML#358, the full {@code
   * fc1 → fc2 → out → softmax} chain narrows along the {@code units} axis ({@code 128 → 256 → 10 →
   * 10}) and every intermediate is registered as a tensor variable. Counts are source-level &mdash;
   * one per distinct value number, deduplicated across calling contexts (wala/ML#371, Option 2)
   * &mdash; so the depth-4 per-caller analysis (train vs. test vs. visualization, wala/ML#379,
   * wala/ML#530) no longer multiplies the count by the number of contexts. The source-level total
   * is 6: parameter {@code x} plus the five registered intermediates of the narrowed chain (one
   * more than the rule-based 4, an extra SSA temporary in the {@code Dense} chain).
   */
  @Test
  public void testNeuralNetwork()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        PythonTensorAnalysisEngine.MODEL_FORWARD_CFA_DEPTH,
        "neural_network.py",
        "NeuralNet.call",
        1,
        6,
        Map.of(3, Set.of(TENSOR_256_784_FLOAT32, TENSOR_10000_784_FLOAT32, TENSOR_5_784_FLOAT32)));
  }

  /**
   * {@code cross_entropy_loss(x, y)} receives logits {@code x} (value 2) and labels {@code y}
   * (value 3). At runtime, {@code x} has shape {@code (256, 10)} dtype {@code float32} and {@code
   * y} has shape {@code (256,)} dtype {@code uint8} (verified by Python assert statements in {@code
   * neural_network.py}).
   *
   * <p>Value 2 flows from {@code pred = neural_net(batch_x, is_training=True)}, which dispatches
   * through {@code Model.__call__} into user-defined {@code NeuralNet.call}. After the fix for
   * wala/ML#358 (chained {@code Dense} shape propagation), value 2 is tracked as a tensor parameter
   * with shape {@code (256, 10) float32} &mdash; the final {@code Dense(num_classes=10)} in the
   * chain narrows to {@code (256, 10)} and that shape flows back through the caller chain.
   *
   * <p>The rule-based tensor variable count is 5 (2 parameters {@code x}, {@code y} + 3
   * intermediate ops {@code cast-to-int64}, {@code sparse_softmax_cross_entropy_with_logits},
   * {@code reduce_mean}). Counts are source-level &mdash; one per distinct value number,
   * deduplicated across calling contexts (wala/ML#371, Option 2) &mdash; so the count is 5, the
   * exact rule-based total, rather than the context-multiplied 10 (value 3 ({@code y}) reached
   * three contexts with a dtype that varies per context; that variation is now captured on the type
   * axis, which unions per vn across contexts, not on the count). The former count-axis duplication
   * tracked by wala/ML#388 is subsumed by this source-level counting.
   */
  @Test
  public void testNeuralNetwork2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        PythonTensorAnalysisEngine.MODEL_FORWARD_CFA_DEPTH,
        "neural_network.py",
        "cross_entropy_loss",
        2,
        5,
        Map.of(2, Set.of(TENSOR_256_10_FLOAT32), 3, Set.of(TENSOR_256_UINT8)));
  }

  /**
   * {@code run_optimization(x, y)} is called with {@code batch_x} and {@code batch_y} from the
   * dataset iteration chain. At runtime, {@code x} has shape {@code (256, 784)} dtype {@code
   * float32} and {@code y} has shape {@code (256,)} dtype {@code uint8} (verified by Python assert
   * statements in {@code neural_network.py}).
   *
   * <p>The test currently fails on value 2's shape only: actual {@code {? of float32}} vs. expected
   * {@code {(256, 784) of float32}}. Dtype routing for slot 0 of the tuple is correct (same state
   * as {@link #testNeuralNetwork()} &mdash; the labels-swap symptom originally reported under
   * wala/ML#396 appears to have been resolved by the per-index delegation work on this branch).
   * What remains is the same shape-propagation gap as {@link #testNeuralNetwork()}: the {@code
   * x_train} chain does not yield a concrete per-index shape through {@code from_tensor_slices} by
   * the time {@code batch_x} reaches {@code run_optimization}'s {@code x}.
   *
   * <p>Rule-based tensor variable count is 6 (2 parameters {@code x}, {@code y} + 4 intermediate
   * ops {@code pred}, {@code loss}, {@code trainable_variables}, {@code gradients}). With the fix
   * for wala/ML#358, {@code pred = neural_net(x, is_training=True)} is now tracked at {@code (256,
   * 10) float32}, bringing the registered count to 4. After wala/ML#430's {@code Gradient}
   * generator, {@code gradients} now registers as one fresh tensor variable (the generator
   * allocates fresh per call rather than aliasing {@code sources}), lifting the count from 4 to 5.
   * {@code trainable_variables} remains a list of tensors and still doesn't register; the residual
   * gap from 5 to 6 is tracked by wala/ML#391.
   */
  @Test
  public void testNeuralNetwork3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        PythonTensorAnalysisEngine.MODEL_FORWARD_CFA_DEPTH,
        "neural_network.py",
        "run_optimization",
        2,
        5,
        Map.of(2, Set.of(TENSOR_256_784_FLOAT32), 3, Set.of(TENSOR_256_UINT8)));
  }

  /**
   * {@code accuracy(y_pred, y_true)} is called from two sites: the training loop with {@code
   * accuracy(pred, batch_y)} where {@code pred} has shape {@code (256, 10)} dtype {@code float32}
   * and {@code batch_y} has shape {@code (256,)} dtype {@code uint8}; and the test-set evaluation
   * with {@code accuracy(pred, y_test)} where {@code pred} has shape {@code (10000, 10)} dtype
   * {@code float32} and {@code y_test} has shape {@code (10000,)} dtype {@code uint8} (verified by
   * Python assert statements in {@code neural_network.py}). The static analysis should union these
   * types for each parameter.
   *
   * <p>Rule-based tensor variable count is 7 (2 parameters {@code y_pred}, {@code y_true} + 5
   * intermediate ops {@code argmax}, {@code cast-to-int64}, {@code equal}, {@code cast-to-float32},
   * {@code reduce_mean}). Counts are source-level &mdash; one per distinct value number,
   * deduplicated across calling contexts (wala/ML#371, Option 2) &mdash; so the count is 7, the
   * exact rule-based total: all five intermediates now register ({@code argmax} via the {@code
   * ReadDataFallback} of wala/ML#437; {@code tf.equal}/{@code tf.cast} no longer drop out under
   * wala/ML#386/wala/ML#387), with the depth-4 per-caller contexts (wala/ML#379, wala/ML#530)
   * deduplicated rather than multiplying the count.
   *
   * <p>Value 2 ({@code y_pred}) is tracked as a tensor parameter after the fix for wala/ML#358
   * (chained {@code Dense} shape propagation): the final {@code Dense(num_classes=10)} in {@code
   * NeuralNet.call} narrows to {@code (256, 10)} and that shape propagates back into {@code
   * accuracy}'s {@code y_pred} parameter. This test runs at k-CFA depth 4 (wala/ML#379) so {@code
   * NeuralNet.call} is analyzed per caller and its layer-output ({@code pred}) no longer collapses
   * the training shape into the test context (wala/ML#530); value 2 is therefore the per-context
   * union {@code {(256, 10) float32, ? float32}}. The test-context contribution is ⊤ shape because
   * the {@code x_test} chain resolves to a rank-preserving but shape-unknown tensor by the time it
   * reaches {@code NeuralNet.call}'s first {@code Dense} operand; closing that gap would narrow the
   * ⊤ to {@code (10000, 10)} (orthogonal to #358/#530).
   */
  @Test
  public void testNeuralNetwork4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // Source-level count is 7 (wala/ML#371, Option 2): the 2 parameters plus the 5
    // intermediate ops, deduplicated across the depth-4 calling contexts. `accuracy()`'s
    // `tf.argmax(...)` is a legitimate tensor source per #380 and now registers via
    // `ReadDataFallback` (#437); under (CGNode, vn) counting its per-context call sites
    // inflated this to 14. Parameter-type expectations (`y_pred`, `y_true`) unchanged.
    test(
        PythonTensorAnalysisEngine.MODEL_FORWARD_CFA_DEPTH,
        "neural_network.py",
        "accuracy",
        2,
        7,
        Map.of(
            2,
            // Per-context union: training call site `accuracy(pred, batch_y)` gives `(256, 10)`;
            // the test-set call site `accuracy(pred, y_test)` resolves to ⊤ shape because the
            // `x_test` chain is shape-unknown by the time it reaches `NeuralNet.call`'s first
            // `Dense` operand. With the depth-4 context separation (wala/ML#530), `argmax`'s result
            // no longer leaks the training shape into the test context. TODO: the test-context ⊤
            // should narrow to `(10000, 10)` once the `x_test` shape gap is closed.
            Set.of(TENSOR_256_10_FLOAT32, TENSOR_UNKNOWN_SHAPE_FLOAT32),
            3,
            Set.of(TENSOR_256_UINT8, TENSOR_10000_UINT8)));
  }

  /**
   * A negative k-CFA depth is invalid (the depth is the call-string length for the targeted context
   * selector) and is rejected at construction (wala/ML#379).
   */
  @Test(expected = IllegalArgumentException.class)
  public void testNegativeTargetedCfaDepthRejected() {
    new PythonTensorAnalysisEngine(
        emptyList(), PythonTensorAnalysisEngine.TENSORFLOW, /* targetedCfaDepth= */ -1);
  }

  /**
   * {@code encoder(x)} receives {@code x} from call sites {@code decoder(encoder(x))} inside {@code
   * run_optimization} (training loop) and {@code decoder(encoder(batch_x))} at the module-level
   * test loop. Both call sites pass batches of shape {@code (256, 784)} dtype {@code float32}
   * (verified by Python assert statements in {@code autoencoder.py}).
   *
   * <p>Expected tensor variable count: 11 &mdash; the distinct SSA vns in {@code encoder}'s body
   * that get tensor types: the parameter {@code x} (vn=2) plus the full layer-1 and layer-2
   * computation: {@code weights['encoder_h1']} (vn=19), {@code biases['encoder_b1']} (vn=24),
   * layer-1 matmul (vn=15), layer-1 add (vn=11), {@code layer_1} sigmoid (vn=4), and the
   * corresponding layer-2 values {@code weights['encoder_h2']} (vn=41), {@code
   * biases['encoder_b2']} (vn=45), layer-2 matmul (vn=38), layer-2 add (vn=35), {@code layer_2}
   * sigmoid (vn=31, returned). Counts are source-level &mdash; one per distinct vn, deduplicated
   * across the 2 call-site contexts (wala/ML#371, Option 2) &mdash; so the two contexts no longer
   * double the total to 22.
   */
  @Test
  public void testAutoencoder()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("autoencoder.py", "encoder", 1, 11, Map.of(2, Set.of(TENSOR_256_784_FLOAT32)));
  }

  /**
   * {@code mean_square(reconstructed, original)} is called only from {@code run_optimization}
   * (itself a FUT &mdash; {@link #testAutoencoder3()}). Its arguments are {@code
   * reconstructed_image = decoder(encoder(x))} and {@code x}, both of which have runtime shape
   * {@code (256, 784)} dtype {@code float32}. Direct call-site asserts aren't possible (they would
   * perturb {@code run_optimization}'s count), so the runtime types are verified indirectly through
   * the {@code batch_x} asserts at the training-loop call of {@code run_optimization}.
   *
   * <p>Expected tensor variable count: 4 — 2 parameters plus 2 intermediate-op tensors picked up by
   * #196's {@code ReadDataFallback}: {@code original - reconstructed} (vn=9) and {@code tf.pow}
   * (vn=13), both flow-through {@code (256, 784) float32}. {@code tf.reduce_mean}'s scalar result
   * is the function's return and isn't tracked as a separate variable. Per-op generators tracked in
   * #449 would tighten the asserted types from ⊤-shape/UNKNOWN-dtype to concrete shapes without
   * changing the count.
   *
   * <p>Value 2 ({@code reconstructed}) resolves to concrete {@code (256, 784) float32} after the
   * {@code tensorflow/python/ops/variables/Variable} allocatable-class declaration was added in
   * {@code tensorflow.xml} (closes <a
   * href="https://github.com/wala/WALA/issues/1889">wala/WALA#1889</a>). With the variable
   * allocations now registered in the heap model, the {@code matmul → add → sigmoid →
   * user-function-return} shape chain fully resolves end-to-end.
   */
  @Test
  public void testAutoencoder2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "autoencoder.py",
        "mean_square",
        2,
        5,
        Map.of(2, Set.of(TENSOR_256_784_FLOAT32), 3, Set.of(TENSOR_256_784_FLOAT32)));
  }

  /**
   * {@code run_optimization(x)} is called from the training loop with {@code batch_x} of shape
   * {@code (256, 784)} dtype {@code float32} (verified by Python assert statements in {@code
   * autoencoder.py}).
   *
   * <p>Expected tensor variable count: 4. Rule-based would be 6 (1 parameter + 5 intermediate ops
   * {@code encoder(x)} result, {@code decoder(...) = reconstructed_image}, {@code mean_square(...)
   * = loss}, {@code trainable_variables}, {@code gradients}), dropping to 4 if list-of-tensors
   * values don't register. After wala/ML#430's {@code Gradient} generator, {@code gradients} now
   * registers as one fresh tensor variable; combined with the previously-registered tensors the
   * count reaches 4 — matching the rule-based-minus-{@code trainable_variables} ceiling. (Pre-#430
   * the count was 1, so the master baseline of 3 had been deliberately preserved as a regression
   * canary; that role is now superseded by reaching the rule-based ceiling.)
   */
  @Test
  public void testAutoencoder3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("autoencoder.py", "run_optimization", 1, 5, Map.of(2, Set.of(TENSOR_256_784_FLOAT32)));
  }

  /**
   * {@code decoder(x)} is called from the same two sites as {@code encoder} ({@link
   * #testAutoencoder()}), but with {@code x} being the output of {@code encoder}. Since {@code
   * encoder}'s layer 2 has dim {@code num_hidden_2 = 64}, {@code decoder} receives {@code (256, 64)
   * float32} (verified by a Python assert in the test loop).
   *
   * <p>Expected tensor variable count: 11 (parallel to {@link #testAutoencoder()}). Same body
   * structure: 11 distinct SSA vns covering the parameter plus the two-layer {@code
   * weights[...]}/{@code biases[...]} / matmul / add / sigmoid chain, counted source-level (one per
   * distinct vn, deduplicated across the 2 call-site contexts; wala/ML#371, Option 2).
   *
   * <p>Value 2 ({@code decoder}'s {@code x} parameter) resolves to concrete {@code (256, 64)
   * float32} after the {@code tensorflow/python/ops/variables/Variable} allocatable-class fix
   * (closes <a href="https://github.com/wala/WALA/issues/1889">wala/WALA#1889</a>). The {@code
   * encoder → decoder} shape chain now flows end-to-end through the XML-summary return keys.
   */
  @Test
  public void testAutoencoder4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("autoencoder.py", "decoder", 1, 11, Map.of(2, Set.of(TENSOR_256_64_FLOAT32)));
  }

  @Test
  public void testSigmoid()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sigmoid.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
  }

  // wala/ML#449 Tier 2: element-wise unary math ops. Each preserves shape and dtype from the
  // input. Same `Sigmoid`-shape pass-through pattern; the receiving function's parameter at
  // vn=2 carries `tf.constant([1.0, 2.0, 3.0])`'s `(3,) float32`.

  @Test
  public void testAbs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_abs.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testAcos()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_acos.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testExp()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_exp.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testExp()} exercising the keyword-argument call site {@code
   * tf.math.exp(x=...)}. {@link com.ibm.wala.cast.python.ml.client.PassThroughUnaryTensorGenerator}
   * routes argument resolution through {@code getArgumentPointsToSet(builder, paramPos, paramName)}
   * which resolves keyword args via {@code paramName}; without a kwarg fixture that branch is
   * dead-on-arrival in the test data.
   */
  @Test
  public void testExpKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_exp_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testTanh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tanh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testRsqrt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_rsqrt.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Companion to {@link #testRsqrt()} exercising the keyword-argument call site. */
  @Test
  public void testRsqrtKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_rsqrt_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Regression guard for wala/ML#449's closing fix: {@code tf.random.truncated_normal(shape)} now
   * dispatches to the {@code TruncatedNormal} generator (via {@code PROPERTY_NAME_GENERATORS}) and
   * resolves to precise {@code (2, 3) float32}. Pre-fix this fell through to {@code
   * ReadDataFallback} and emitted {@code ⊤ shape / UNKNOWN dtype}.
   */
  @Test
  public void testTruncatedNormal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_truncated_normal.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testLogSoftmax()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_log_softmax.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testLogSoftmax()} exercising the keyword-argument call site {@code
   * tf.nn.log_softmax(logits=...)}.
   */
  @Test
  public void testLogSoftmaxKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_log_softmax_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testL2Normalize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_l2_normalize.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testSigmoid2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sigmoid2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
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
   * Regression test for wala/ML#435: a recursive Python function whose return value flows back into
   * itself used to drive {@link
   * com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory#getGenerator(
   * com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable,
   * com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder)} into unbounded recursion
   * via the return-value follow-through and the assignment-graph predecessor walk, ending in {@code
   * StackOverflowError}. The cycle guard added in this PR returns {@code null} when a {@code
   * PointsToSetVariable} is re-encountered along the current call chain. With the cycle guard in
   * place, the recursive call's return value still resolves through its base-case branch — the
   * input {@code tf.constant(1)} (a scalar int32 tensor) flows back to {@code f}'s parameter.
   *
   * <p>The Python test deliberately omits the {@code @tf.function} decorator. Empirically, the
   * regression reproduces without it (verified by reverting the cycle guard locally — this test
   * still SOes), and the decorated form would re-trace the recursive call at runtime and hit
   * Python's recursion limit before the assertions could run. The undecorated form lets {@code
   * python3.10} execute the file to completion with the {@code shape}/{@code dtype} assertions on
   * {@code result} exercised.
   */
  @Test
  public void testRecursiveFunction()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_recursive_function.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Regression test for wala/ML#451 (reproducer 1): a recursive Python function whose only call
   * sites are {@code recursive_fn(5)} (a Python int literal) and {@code recursive_fn(n - 1)} (still
   * a Python int) must not classify its parameter as a tensor. There is no {@code tf.constant}, no
   * decorator, and no tensor anywhere in the program — the analysis should report zero tensor
   * parameters and zero function-local tensor variables.
   */
  @Test
  public void testRecursionIntOnly()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_recursion_int_only.py", "recursive_fn", 0, 0);
  }

  /**
   * Regression test for wala/ML#451 (reproducer 2): a recursive function whose external call site
   * passes a real tensor ({@code recursive_fn(tf.constant(5))}). The parameter {@code n} should be
   * classified as a scalar int32 tensor (the {@code tf.constant} flows through the assignment graph
   * from the caller), and {@code n - 1} inside the body is a tensor binop too — the binop
   * operand-tensor gate in {@link TensorGeneratorFactory} dispatches {@code n - 1} to {@link
   * ElementWiseOperation} because at least one operand ({@code n}) has tensor evidence in its PTS.
   *
   * <p>Tensor-variable count breakdown: {@code vn=2} (parameter) is seen across two analysis
   * contexts (the top-level call and the recursive self-call), and {@code vn=10} ({@code n - 1}) is
   * the binop result in the top-level context. Counts are source-level &mdash; one per distinct
   * value number, deduplicated across calling contexts (wala/ML#371, Option 2) &mdash; so the
   * parameter's two contexts collapse and the count is 2 ({@code vn=2} and {@code vn=10}), not the
   * three {@code (CGNode, vn)} entries the analysis registers.
   */
  @Test
  public void testRecursionTensorOnly()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_recursion_tensor_only.py",
        "recursive_fn",
        1,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testSliceOpaque()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_slice_opaque.py", "consume", 0, 0);
  }

  @Test
  public void testSliceOpaqueIter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_slice_opaque_iter.py", "consume", 0, 0);
  }

  /**
   * Regression test for <a href="https://github.com/wala/ML/issues/653">wala/ML#653</a>: Python
   * list repetition ({@code [0] * 3}) is not a tensor. The {@code *} binop has a {@code list}
   * operand and an {@code int} operand, so it is list repetition (producing a list), not tensor
   * scalar-multiplication. The binop operand-tensor gate must not treat the bare {@code list}
   * operand as tensor evidence, so {@code consume}'s parameter is not classified as a tensor.
   */
  @Test
  public void testListRepetition()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_list_repetition.py", "consume", 0, 0);
  }

  /**
   * Regression test for wala/ML#451 (reopen): asserts the underlying PA state that Hybridize's
   * {@code Function.inferPrimitiveParameters} consumes &mdash; specifically, that no primitive
   * {@link ConstantKey} is reachable through the parameter's points-to set when traversed
   * transitively through instance fields. The traversal here mirrors the recursion in {@code
   * Function.containsPrimitive(InstanceKey, ...)}: a {@code ConstantKey} with a non-null value
   * (excluding bools) is "primitive"; an {@link AllocationSiteInNode} or {@link ConcreteTypeKey} is
   * recursively examined through its declared instance fields.
   *
   * <p>The fixture is the same as {@link #testRecursionTensorOnly()} ({@code recursive_fn(tf
   * .constant(5))}). Pre-fix, this assertion failed because {@code tensorflow.xml}'s {@code
   * tf.constant.do} method bound the user's {@code value} argument to the alloc's {@code value}
   * field via {@code <putfield>}, so the field-traversal walk found the user's {@code
   * ConstantKey<Integer:5>} and classified the parameter as primitive even though the alloc IS a
   * tensor producer. The XML now omits that binding (the {@link
   * com.ibm.wala.cast.python.ml.client.Constant} generator reads dtype/shape directly from the
   * call's value-arg PTS rather than from the alloc's field), and a CG-walk fallback in {@code
   * TensorGenerator.getShapesFromShapeArgument} keeps shape inference working for cases like {@code
   * tf.constant([2, 3])} as a shape argument.
   */
  @Test
  public void testRecursionTensorOnlyHasNoPrimitiveInPTS()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(emptyList(), new String[] {"tf2_test_recursion_tensor_only.py"});
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);
    engine.performAnalysis(builder);

    String functionSignature = "script tf2_test_recursion_tensor_only.py.recursive_fn.do()LRoot;";
    boolean checkedAtLeastOneContext = false;

    for (CGNode node : CG) {
      if (!node.getMethod().getSignature().equals(functionSignature)) continue;
      // Parameter `n` is at vn=2 (vn=1 is `self`/function object).
      PointerKey paramPK =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
      OrdinalSet<InstanceKey> pts = builder.getPointerAnalysis().getPointsToSet(paramPK);
      if (pts == null || pts.isEmpty()) continue;
      checkedAtLeastOneContext = true;
      for (InstanceKey ik : pts) {
        Set<InstanceKey> seen = new HashSet<>();
        assertTrue(
            "Parameter `n` of recursive_fn(tf.constant(5)) should not have any primitive"
                + " ConstantKey reachable through PA field traversal in context "
                + node.getContext()
                + " (instance="
                + ik
                + "). This is the underlying state that Hybridize's"
                + " Function.containsPrimitive consumes (wala/ML#451 reopen).",
            !containsPrimitiveByFieldWalk(ik, builder.getPointerAnalysis(), seen));
      }
    }
    assertTrue(
        "Expected to check at least one CGNode/context for recursive_fn with non-empty PTS"
            + " for vn=2; if this assertion fails, the test setup may not have produced any"
            + " analyzable parameter.",
        checkedAtLeastOneContext);
  }

  /**
   * Regression guard for wala/ML#657: a {@code tf.keras.Model} subclass reached via {@code
   * model(x)} callable dispatch keeps its call-graph node even when another module defines a
   * same-named {@code class Model}. The subclass uses the ubiquitous bare-import idiom {@code from
   * tensorflow.keras import Model; class MyModel(Model)}; when a second module ({@code
   * tf2_657_collide.py}) also defines {@code class Model}, the bare base name previously
   * mis-resolved across modules, so {@code MyModel}'s superclass came back {@code null} and {@code
   * MyModel.call} was dropped from the call graph (which is the empty {@code getNodes(...)} that
   * fails Hybridize's side-effect, recursion, and primitive-parameter preconditions). The fix falls
   * back to {@code object} when the base name has no local resolution, matching the collision-free
   * case.
   */
  @Test
  public void testModelSubclassNameCollision()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(emptyList(), new String[] {"tf2_657_model_call.py", "tf2_657_collide.py"});
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);
    assertTrue(
        "MyModel.call should have a call-graph node despite a colliding `class Model` in another"
            + " module (wala/ML#657).",
        CG.stream()
            .anyMatch(
                n ->
                    n.getMethod()
                        .getSignature()
                        .contains("tf2_657_model_call.py.MyModel.call.do")));
  }

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
  private static boolean containsPrimitiveByFieldWalk(
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
  public void testZerosLikeTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_zeros_like_tensor.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
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
  public void testInputWithBatchSize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t1 = TensorType.of(FLOAT_32, 16, 32);
    TensorType t2 = TensorType.of(FLOAT_32, 5, 10, 10);
    TensorType t3 = new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(5)));

    test(
        "tf2_test_input_batch_size.py",
        "check_input",
        3,
        3,
        Map.of(2, Set.of(t1), 3, Set.of(t2), 4, Set.of(t3)));
  }

  @Test
  public void testInputInt32()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t1 = TensorType.of(INT_32, 32, 10);
    TensorType t2 = TensorType.of(INT_32, 8, 5, 5);

    test("tf2_test_input_int32.py", "check_input", 2, 2, Map.of(2, Set.of(t1), 3, Set.of(t2)));
  }

  @Test
  public void testInputMixedArgs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // input1: shape=(32, 10), dtype=int32
    TensorType t1 = TensorType.of(INT_32, 32, 10);
    // input2: shape=(16, 5, 5), dtype=float32
    TensorType t2 = TensorType.of(FLOAT_32, 16, 5, 5);
    // input3: shape=(None, 20), dtype=int32
    TensorType t3 = new TensorType(INT_32, asList(DynamicDim.INSTANCE, new NumericDim(20)));

    test(
        "tf2_test_input_mixed_args.py",
        "check_input",
        3,
        3,
        Map.of(2, Set.of(t1), 3, Set.of(t2), 4, Set.of(t3)));
  }

  /**
   * The `tensor` parameter wraps an existing tensor, so the result takes that tensor's shape and
   * dtype verbatim, with no batch dimension prepended (wala/ML#617).
   */
  @Test
  public void testInputTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = TensorType.of(FLOAT_32, 2, 3);

    test("tf2_test_input_tensor_kw.py", "check_input", 1, 1, Map.of(2, Set.of(t)));
    test("tf2_test_input_tensor_pos.py", "check_input", 1, 1, Map.of(2, Set.of(t)));
  }

  /**
   * A ragged `Input` has the same tracked shape and dtype as a dense one, so the `ragged` parameter
   * is modeled by treating it as transparent to shape and dtype inference (wala/ML#617).
   */
  @Test
  public void testInputRagged()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(10)));

    test("tf2_test_input_ragged_kw.py", "check_input", 1, 1, Map.of(2, Set.of(t)));
  }

  /**
   * The `type_spec` parameter supplies the full type, so the result takes the spec's shape and
   * dtype verbatim, with no batch dimension prepended (wala/ML#617).
   */
  @Test
  public void testInputTypeSpec()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = new TensorType(INT_32, asList(DynamicDim.INSTANCE, new NumericDim(4)));

    test("tf2_test_input_type_spec_kw.py", "check_input", 1, 1, Map.of(2, Set.of(t)));
  }

  /**
   * A sparse `Input` has the same logical shape and dtype as a dense one, so the `sparse` parameter
   * is modeled by treating it as transparent to shape and dtype inference (wala/ML#616).
   */
  @Test
  public void testInputSparse()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(10)));

    test("tf2_test_input_sparse_kw.py", "check_input", 1, 1, Map.of(2, Set.of(t)));
    test("tf2_test_input_sparse_pos.py", "check_input", 1, 1, Map.of(2, Set.of(t)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>: a custom
   * {@code fit} iterates an {@code experimental_distribute_dataset}-wrapped {@code tf.data} dataset
   * and threads the yielded {@code (inputs, targets)} tuple into {@code train_step}. The strategy's
   * {@code experimental_distribute_dataset} is a pass-through of its dataset argument, so the
   * distributed dataset stays a recognized tensor iterable and {@code train_step}'s {@code inputs}
   * and {@code targets} parameters type to {@code (2,)} float32 rather than being dropped (which
   * cascaded to every downstream consumer).
   */
  @Test
  public void testDistributeFitTupleParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_distribute_fit_tuple_param.py",
        "Model.train_step",
        2,
        3,
        Map.of(3, Set.of(TensorType.of(FLOAT_32, 2)), 4, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/623">wala/ML#623</a>: a {@code
   * padded_batch} element threaded through a custom {@code fit} into {@code train_step} types the
   * parameters. {@code padded_batch} was unmodeled, so the per-element tensor type was dropped
   * before reaching the callee; modeling it like {@code batch} recovers it. The two parameters type
   * to {@code (2, 2)} int32 (the batch dimension prepended to the {@code (2,)} element).
   */
  @Test
  public void testPaddedBatchParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = TensorType.of(INT_32, 2, 2);

    test(
        "tf2_test_padded_batch_param.py",
        "Model.train_step",
        2,
        3,
        Map.of(3, Set.of(t), 4, Set.of(t)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>'s gpt-2
   * case: a callee parameter that receives a tensor argument at a direct method-call site is typed
   * by Ariadne. The {@code Gpt2} model, dataset pipeline, and {@code _train_step}/{@code
   * train_step}/{@code get_loss} dispatch are vendored verbatim from {@code
   * akanyaani/gpt-2-tensorflow2.0} ({@code gpt2_model.py}, {@code data_pipeline.py}); only the
   * transformer {@code layers} (and the {@code utils}/{@code scripts} helpers) are stubbed to
   * pass-throughs, since {@code get_loss}'s parameter typing does not depend on the model body.
   *
   * <p>A {@code padded_batch} dataset element flows through {@code fit} to the {@code
   * @tf.function(input_signature=...)}-decorated {@code train_step}, then {@code _train_step}, then
   * {@code get_loss(targets, predictions)}. The {@code real} parameter (vn=3), bound to the
   * dataset-sourced {@code targets}, types to {@code (2, 2)} int32, so Ariadne emits the parameter
   * type for this exact shape. With wala/ML#665 forwarding wildcard import bindings, {@code
   * pred} types too: the stubbed model body's forward output is a rank-3 union with the vocab
   * dimension recovered. This pins that wala/ML#618's residual gpt-2 failure is downstream of
   * Ariadne, not an emission gap.
   */
  @Test
  public void testGpt2InterprocGetLoss()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_proj/layers/__init__.py", "gpt2_proj/layers/embedding_layer.py",
          "gpt2_proj/layers/feed_forward.py", "gpt2_proj/layers/layer_norm.py",
          "gpt2_proj/layers/attention_layer.py", "gpt2_proj/utils/__init__.py",
          "gpt2_proj/utils/tf_utils.py", "gpt2_proj/scripts/__init__.py",
          "gpt2_proj/scripts/utils.py", "gpt2_proj/data_pipeline.py",
          "gpt2_proj/gpt2_model.py", "gpt2_proj/tf2_test_gpt2_probe.py"
        },
        "gpt2_model.py",
        "Gpt2.get_loss",
        "gpt2_proj",
        2,
        8,
        Map.of(
            3,
            Set.of(TensorType.of(INT_32, 2, 2)),
            4,
            // The model forward output: rank 3 with the vocab dimension recovered as the
            // constant 100; the dtype is refinable once `add_weight` consumes its `dtype`
            // argument. The (2, 2) int32 member is the call-site union with the label tensor.
            Set.of(
                new TensorType(
                    UNKNOWN, asList(DynamicDim.INSTANCE, DynamicDim.INSTANCE, new NumericDim(100))),
                new TensorType(
                    UNKNOWN,
                    asList(new SymbolicDim("?"), new SymbolicDim("?"), new SymbolicDim("?"))),
                TensorType.of(INT_32, 2, 2))));
  }

  /**
   * The full subject from {@code akanyaani/gpt-2-tensorflow2.0} (a perf-eval corpus subject)
   * vendored verbatim with the REAL transformer layers and {@code input_fn}, unlike the stubbed
   * {@link #testGpt2InterprocGetLoss()}. {@code get_loss}'s {@code real} now types end to end
   * (wala/ML#618), resolving the whole dataset-sourced chain that previously dehybridized it.
   *
   * <p>The chain: {@code real} (the dataset-sourced {@code targets}) is built in {@code input_fn}
   * as {@code
   * tf.data.TFRecordDataset(...).map(parse_example).padded_batch(...).repeat(...).prefetch(...)},
   * passed as a list {@code _model.fit([_train, _test], ...)}, list-unpacked, iterated with {@code
   * enumerate} and nested unpacking, and forwarded through an indirected {@code train_fuc} into
   * {@code train_step} → {@code _train_step} → {@code get_loss}. Each layer is modeled: {@code
   * parse_example} densifies a {@code tf.io.VarLenFeature} through a dict to {@code (?,)} int32
   * (wala/ML#646, pinned by {@link #testParseExampleTuple()}); {@code map} types the element from
   * {@code map_func}'s return (wala/ML#506, {@link #testDatasetMapTuple()}); a pass-through
   * transform after {@code map} keeps the mapped type (wala/ML#649, {@link
   * #testDatasetMapRepeat()}); {@code TFRecordDataset} is chainable ({@link #testTfrecordMap()});
   * the dataset survives the list (wala/ML#648, {@link #testFitLoop()}); and the {@code
   * padded_batch} dims apply (wala/ML#673). So {@code real} resolves to the batched element {@code
   * (32, ?)} int32 (the pipeline's {@code batch_size=32} with the pad-to-longest sequence dim),
   * unioned with the standard partial-batch sibling {@code (?, ?)}.
   *
   * <p>{@code pred} types too (wala/ML#665): the model forward output is a tensor union. With
   * {@code add_weight} consuming its {@code shape}/{@code dtype} arguments (wala/ML#667) and
   * constructor keyword arguments forwarded to {@code __init__} (wala/ML#664), the runtime-true
   * vocab dimension is concrete ({@code (?, ?, 10)}). Receiver-keyed trampoline contexts
   * (wala/ML#679) removed the spurious {@code (?, ?, 4)} constructor-collapse member, but the
   * flat-logits matmul member that carried the embedding dimension ({@code (?, 8)} float32)
   * degrades to {@code ? of float32} in decoder-stack propagation. The union is the
   * order-independent fixed point (wala/ML#674): identical across runs and across suite/single-test
   * modes. Analyzed statically here, like the consumer's vendoring; it runs in the perf-eval with
   * its tfrecord/data setup.
   *
   * <p>TODO: Expect the float32 member's concrete shape back once <a
   * href="https://github.com/wala/ML/issues/682">wala/ML#682</a> recovers concrete shapes through
   * the decoder stack under receiver-keyed contexts.
   */
  @Test
  public void testGpt2GetLossVendored()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_vendored/layers/__init__.py", "gpt2_vendored/layers/embedding_layer.py",
          "gpt2_vendored/layers/feed_forward.py", "gpt2_vendored/layers/layer_norm.py",
          "gpt2_vendored/layers/attention_layer.py", "gpt2_vendored/utils/__init__.py",
          "gpt2_vendored/utils/tf_utils.py", "gpt2_vendored/scripts/__init__.py",
          "gpt2_vendored/scripts/utils.py", "gpt2_vendored/data_pipeline.py",
          "gpt2_vendored/A.py"
        },
        "A.py",
        "Gpt2.get_loss",
        "gpt2_vendored",
        2,
        8,
        Map.of(
            3,
            Set.of(new TensorType(INT_32, asList(DynamicDim.INSTANCE))),
            4,
            Set.of(
                new TensorType(
                    UNKNOWN, asList(DynamicDim.INSTANCE, DynamicDim.INSTANCE, new NumericDim(10))),
                TENSOR_UNKNOWN_SHAPE_FLOAT32,
                new TensorType(
                    UNKNOWN,
                    asList(new SymbolicDim("?"), new SymbolicDim("?"), new SymbolicDim("?"))))));
  }

  /**
   * Regression guard for wala/ML#618's data-pipeline fix: {@code tf.sparse.to_dense} of a {@code
   * SparseTensor} types its dense result from the operand's {@code dense_shape} field (shape) and
   * {@code values} field (dtype), so {@code consume}'s parameter types to {@code (2,2)} int32.
   * Modeling this is what un-strands {@code get_loss}'s {@code real} (the dataset-sourced {@code
   * targets}) in {@link #testGpt2GetLossVendored()}.
   */
  @Test
  public void testSparseToDense()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_to_dense.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 2, 2))));
  }

  /**
   * Regression guard for wala/ML#645: {@code tf.io.VarLenFeature(dtype)} models the SparseTensor a
   * variable-length feature parses to, so {@code tf.sparse.to_dense} of it types from the feature's
   * dtype ({@code int64}) and the API-contract shape (rank-1 with a dynamic length, {@code (?,)}).
   * The {@code io}-registration fix makes {@code tf.io.*} resolve at all (they were registered
   * under {@code tf}, not the {@code io} object). The rank-1 dynamic shape is the contract-model
   * refinement of wala/ML#647 (formerly ⊤).
   */
  @Test
  public void testVarLenFeature()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_var_len_feature.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_DYNAMIC_INT64)));
  }

  /**
   * The realistic gpt-2 shape for wala/ML#645: a {@code VarLenFeature} in a feature dict, parsed by
   * {@code tf.io.parse_single_example}, subscripted, and densified. {@code consume}'s parameter
   * types to {@code (?,)} int64: the VarLenFeature SparseTensor now keeps a live points-to set
   * through the dict {@code putfield}/{@code getfield} (wala/ML#646), and {@code
   * tf.sparse.to_dense} resolves the dict-routed operand by dispatching the {@code VarLenFeature}
   * generator at the SparseTensor's allocation site, recovering the feature's dtype and the rank-1
   * dynamic (contract) shape. The static shape is {@code (?,)}, not the concrete {@code (2,)} the
   * Python runtime produces, because the example's length is lost across the serialize/parse
   * round-trip. This is the dict-routed companion to the direct {@link #testVarLenFeature()};
   * together they un-strand {@code get_loss}'s {@code real} in {@link #testGpt2GetLossVendored()}.
   */
  @Test
  public void testParseSingleExample()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_parse_single_example.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_DYNAMIC_INT64)));
  }

  /**
   * Mirrors gpt-2's {@code parse_example} for wala/ML#618: a tuple return over two {@code
   * tf.cast(tf.sparse.to_dense(parsed[k]), tf.int32)} values, each parsed from a {@code
   * VarLenFeature} in a feature dict, but called directly (no {@code dataset.map}). The recovered
   * {@code (?,)} int64 propagates through {@code to_dense}, the int32 cast, the tuple return, and
   * the destructuring, so {@code consume}'s parameter ({@code targets}) types to {@code (?,)}
   * int32. Together with {@link #testParseSingleExample()} this isolates the {@code dataset.map}
   * element-type layer as the sole remaining gap for the full {@link #testGpt2GetLossVendored()}
   * subject.
   */
  @Test
  public void testParseExampleTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_parse_example_tuple.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_32, asList(DynamicDim.INSTANCE)))));
  }

  /**
   * Regression guard for wala/ML#506: {@code tf.data.Dataset.map(map_func)} types its elements from
   * {@code map_func}'s return, not the receiver's elements. {@code map_func} here ({@code double})
   * consumes its argument ({@code tf.cast(x, tf.int64)}), exercising both halves: the callback's
   * parameter resolves to the upstream element type ({@code (4,)} float32, so the cast's shape is
   * {@code (4,)}) and its return ({@code (4,)} int64) becomes the mapped element type, recovered at
   * {@code consume}.
   */
  @Test
  public void testDatasetMap()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset_map.py", "consume", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 4))));
  }

  /**
   * Companion to {@link #testDatasetMap()} for a tuple-returning {@code map_func}: {@code split}
   * returns {@code (a, b)}, and iterating {@code for x, y in ds} resolves {@code y} (the second
   * component) to {@code (4,)} int64 via the mapped dataset's per-index tuple typing. wala/ML#506.
   */
  @Test
  public void testDatasetMapTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset_map_tuple.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 4))));
  }

  /**
   * Like {@link #testDatasetMapTuple()} but iterated with {@code enumerate} and nested unpacking
   * ({@code for i, (x, y) in enumerate(ds)}) — the gpt-2 {@code fit}-loop shape (wala/ML#618). The
   * mapped tuple element's second component {@code y} still resolves to {@code (4,)} int64.
   * wala/ML#506.
   */
  @Test
  public void testDatasetMapEnumerate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset_map_enumerate.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 4))));
  }

  /**
   * Regression guard for wala/ML#648: a {@code tf.data.Dataset} stored in a list and read back
   * ({@code [ds, ds][0]}) keeps its element type when iterated. The element lowers to {@code
   * d[iterator]}, a property read whose receiver came from a list getfield, so the def-chain alone
   * resolves the list, not the dataset; the fix recovers the dataset from the element at the
   * constant index and seeds the read off the receiver's points-to set. {@code consume}'s parameter
   * types to the sliced element {@code (4,)} float32.
   */
  @Test
  public void testDatasetInList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset_in_list.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The gpt-2 {@code fit}-side shape for wala/ML#618, end to end: a mapped dataset passed in a list
   * ({@code fit([ds, ds])}), list-unpacked, iterated with {@code enumerate} and nested unpacking,
   * then forwarded through an indirected callback into {@code get_loss}. {@code real} (the second
   * mapped tuple component) types to {@code (4,)} int64, exercising wala/ML#648 (dataset through a
   * list) together with wala/ML#506 (the {@code map} tuple element). The full vendored subject
   * ({@link #testGpt2GetLossVendored()}) additionally chains {@code padded_batch}/{@code repeat}/
   * {@code prefetch} behind an {@code input_fn} return, which still loses the transformation chain.
   */
  @Test
  public void testFitLoop()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_fit_loop.py", "consume", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 4))));
  }

  /**
   * Regression guard for wala/ML#649: a pass-through transform after {@code map} keeps the mapped
   * element type. {@code map(split).repeat(2)} yields the same {@code (int32, int64)} tuple
   * elements as {@code map(split)}, so {@code y} types to {@code (4,)} int64. Before the fix,
   * {@code repeat}'s receiver-inheritance resolved the {@code map} receiver to a plain {@code
   * DatasetGenerator} (inheriting the upstream base), dropping {@code map_func}'s return; it now
   * resolves to a {@code DatasetMapGenerator} reading the {@code element} field off the receiver
   * instance.
   */
  @Test
  public void testDatasetMapRepeat()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset_map_repeat.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 4))));
  }

  /**
   * Regression guard for wala/ML#618: {@code tf.data.TFRecordDataset(...)} is a chainable dataset,
   * so {@code .map(parse_example)} resolves and the VarLenFeature-parsed {@code targets} types to
   * {@code (?,)} int32. Previously {@code TFRecordDataset} was a bare {@code Dataset} field with no
   * {@code do()}, so the chain did not resolve.
   */
  @Test
  public void testTfrecordMap()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tfrecord_map.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_32, asList(DynamicDim.INSTANCE)))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>: {@code
   * strategy.run(fn, (a, b))} forwards both elements of the positional {@code args} tuple into the
   * two-parameter callback {@code step_fn(inp, tar)}, not just the first. Both {@code
   * consume_inp}'s and {@code consume_tar}'s parameters type to {@code (2,)} int32; previously the
   * {@code tensorflow/distribute/run/run} model forwarded only {@code args[0]}, so {@code tar} (and
   * {@code consume_tar}'s parameter) stayed untyped.
   */
  @Test
  public void testStrategyRunTwoArgsInp()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_strategy_run_two_args.py",
        "consume_inp",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 2))));
  }

  /**
   * Companion to {@link #testStrategyRunTwoArgsInp()} pinning the <em>second</em> forwarded
   * element: {@code consume_tar}'s parameter types to {@code (2,)} int32, confirming {@code
   * strategy.run} forwards {@code args[1]}. See <a
   * href="https://github.com/wala/ML/issues/618">wala/ML#618</a>.
   */
  @Test
  public void testStrategyRunTwoArgsTar()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_strategy_run_two_args.py",
        "consume_tar",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 2))));
  }

  /**
   * Companion to {@link #testGpt2InterprocGetLoss()} that drives the <em>distributed</em> reach of
   * <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>'s gpt-2 case: the same vendored
   * {@code Gpt2} model, but reached via {@code distributed_train_step} &rarr; {@code
   * _distributed_train_step} &rarr; {@code mirrored_strategy.run(step_fn, args=(inputs, targets))}
   * &rarr; {@code step_fn} &rarr; {@code get_loss(tar, logits)}, the path the real subject takes.
   *
   * <p>{@code get_loss}'s {@code real} parameter (vn=3), bound to the dataset-sourced {@code
   * targets} that flows through {@code strategy.run}'s {@code args} tuple into {@code step_fn}'s
   * {@code tar}, types to {@code (2, 2)} int32 exactly as in the direct reach. This exercises both
   * halves of the wala/ML#618 distributed-reach fix: the {@code tensorflow/distribute/run/run}
   * model forwarding both tuple elements (see {@link #testStrategyRunTwoArgsInp()}), and the {@code
   * args} parameter name surviving summary loading (the <a
   * href="https://github.com/wala/WALA/pull/1972">wala/WALA#1972</a> fix to {@code
   * XMLMethodSummaryReader}'s name filter), without which the keyword {@code args=} could not bind.
   */
  @Test
  public void testGpt2DistributedGetLoss()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_proj/layers/__init__.py", "gpt2_proj/layers/embedding_layer.py",
          "gpt2_proj/layers/feed_forward.py", "gpt2_proj/layers/layer_norm.py",
          "gpt2_proj/layers/attention_layer.py", "gpt2_proj/utils/__init__.py",
          "gpt2_proj/utils/tf_utils.py", "gpt2_proj/scripts/__init__.py",
          "gpt2_proj/scripts/utils.py", "gpt2_proj/data_pipeline.py",
          "gpt2_proj/gpt2_model.py", "gpt2_proj/tf2_test_gpt2_distributed_probe.py"
        },
        "gpt2_model.py",
        "Gpt2.get_loss",
        "gpt2_proj",
        2,
        8,
        Map.of(
            3,
            Set.of(TensorType.of(INT_32, 2, 2)),
            4,
            // The model forward output: rank 3 with the vocab dimension recovered as the
            // constant 100; the dtype is refinable once `add_weight` consumes its `dtype`
            // argument. The (2, 2) int32 member is the call-site union with the label tensor.
            Set.of(
                new TensorType(
                    UNKNOWN, asList(DynamicDim.INSTANCE, DynamicDim.INSTANCE, new NumericDim(100))),
                new TensorType(
                    UNKNOWN,
                    asList(new SymbolicDim("?"), new SymbolicDim("?"), new SymbolicDim("?"))),
                TensorType.of(INT_32, 2, 2))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/598">wala/ML#598</a>: a bare
   * {@code numpy.array} value propagates its {@code TensorType} to a callee parameter. The issue's
   * reproducer; {@code f}'s parameter types to {@code (3,)} {@code float64}: {@code NpArray} infers
   * the list-literal shape, and the dtype from numpy's promotion of the Python float literals (<a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>). The runtime dtype is {@code
   * float64} (numpy promotes Python {@code float} to {@code float64}, not the {@code float32}
   * TF-literal convention).
   */
  @Test
  public void testNpArrayBareParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nparray_param.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 3))));
  }

  /**
   * Companion to {@link #testNpArrayBareParam()} for the integer-promotion path of <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>: a bare {@code numpy.array} of
   * Python ints types to {@code (3,)} {@code int64}, because numpy promotes Python {@code int} to
   * {@code int64} (not the {@code int32} TF-literal convention).
   */
  @Test
  public void testNpArrayIntParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nparray_int_param.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 3))));
  }

  /**
   * Companion to {@link #testNpArrayBareParam()} for the boolean path of <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>: an all-boolean literal array
   * types to {@code (2,)} {@code bool}.
   */
  @Test
  public void testNpArrayBoolParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nparray_bool_param.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(BOOL, 2))));
  }

  /**
   * Companion to {@link #testNpArrayBareParam()} for the string path of <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>: an all-string literal array types
   * to {@code (2,)} {@code string} (a string leaf subsumes the array in numpy's promotion).
   */
  @Test
  public void testNpArrayStringParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nparray_string_param.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(STRING, 2))));
  }

  /**
   * Companion to {@link #testNpArrayBareParam()} for the nested-literal promotion path of <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>: a nested literal mixing ints and
   * a float promotes to {@code (2, 2)} {@code float64}, exercising the walk's descent through
   * nested lists and the float-over-int promotion.
   */
  @Test
  public void testNpArrayNestedParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nparray_nested_param.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 2, 2))));
  }

  /**
   * Companion to {@link #testNpArrayBareParam()} for the complex path of <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>: an all-complex literal array
   * types to {@code (2,)} {@code complex128} (numpy promotes Python {@code complex} to {@code
   * complex128}).
   */
  @Test
  public void testNpArrayComplexParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nparray_complex_param.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(COMPLEX_128, 2))));
  }

  /**
   * Companion to {@link #testNpArrayBareParam()} for the non-literal-source floor of <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>: when {@code x} is itself an
   * {@code np.ndarray} rather than a Python literal, numpy preserves the source's dtype, which the
   * walk does not model, so it floors to ⊤. The nested-array shape does not propagate through the
   * outer {@code np.array} either, so both axes are ⊤.
   */
  @Test
  public void testNpArrayArraySource()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nparray_array_source.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/598">wala/ML#598</a>: the
   * {@code tf.constant}-wrapped {@code numpy.array} form also propagates to the callee parameter.
   * It currently types to {@code ⊤} unknown, coarser than the bare form's {@code (3,)} because
   * {@code tf.constant} drops the array shape.
   *
   * <p>TODO: This pins the current imprecise shape. Once <a
   * href="https://github.com/wala/ML/issues/625">wala/ML#625</a> lands, the parameter should type
   * to {@code (3,)} {@code float64} (the bare form's result now that <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a> models numpy dtype promotion), and
   * this assertion should be updated accordingly.
   */
  @Test
  public void testNpArrayWrappedParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nparray_wrapped_param.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>: a tensor
   * passed interprocedurally to a callee types the callee's parameter. {@code Model.get_loss}'s
   * {@code real} and {@code pred} receive {@code tf.constant} tensors via {@code train_step}, so
   * both type to {@code (3,)} float32 rather than being missed.
   */
  @Test
  public void testInterprocTensorParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = TensorType.of(FLOAT_32, 3);

    test(
        "tf2_test_interproc_tensor_param.py",
        "Model.get_loss",
        2,
        4,
        Map.of(3, Set.of(t), 4, Set.of(t)));
  }

  /**
   * A {@code @tf.function} without {@code input_signature} creates {@code tf.constant(5)} and
   * passes it to {@code g} (the FUT), so {@code g}'s parameter is that scalar. At runtime {@code g}
   * receives {@code ()} int32 and Ariadne agrees: a positive guard that a value flowing through a
   * decorated body to a callee types the callee's parameter correctly.
   */
  @Test
  public void testDecoratedCallDepth()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_decorated_call_depth.py", "g", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32))));
  }

  /**
   * A {@code @tf.function(input_signature=[(None,) int32])} passes its parameter to {@code g} (the
   * FUT). What {@code g} receives depends on the execution mode, which a static analysis cannot
   * determine: traced (the default) the signature governs and {@code g} receives {@code (None,)}
   * int32; under {@code run_functions_eagerly} the signature is ignored and {@code g} receives the
   * call-site argument's {@code (3,)} int32. So the sound type of {@code g}'s parameter is the set
   * {@code {(None,), (3,)}} int32.
   *
   * <p>TODO: this pins the current behavior. Ariadne does not consume {@code input_signature}, so
   * it produces only the argument-derived {@code (3,)} element and misses the signature-derived
   * {@code (None,)} one; the sound result is the set {@code {(None,), (3,)}} int32. Tracked by <a
   * href="https://github.com/wala/ML/issues/638">wala/ML#638</a>.
   */
  @Test
  public void testDecoratedCallDepthInputSig()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_decorated_call_depth_input_sig.py",
        "g",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/637">wala/ML#637</a>: a {@code
   * tf.constant} with an explicit {@code dtype=tf.complex64} types to {@code complex64} (now
   * modeled by the {@code DType} enum) rather than ⊤, so {@code consume}'s parameter is {@code
   * (2,)} complex64.
   */
  @Test
  public void testConstantComplex64()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_complex64.py", "consume", 1, 1, Map.of(2, Set.of(TensorType.of(COMPLEX_64, 2))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/642">wala/ML#642</a>: a
   * faithful copy of <a href="https://github.com/wala/ML/issues/637">wala/ML#637</a>'s example,
   * {@code tf.constant([1 + 2j, 3 + 4j], dtype=tf.complex64)}. The complex literal ({@code 2j})
   * folds to a {@code PyComplex} whose {@code asInt()} raises a TypeError; before wala/ML#642 that
   * uncaught exception aborted the module's front-end translation and emptied its entrypoint set.
   * The front end now skips folding it, so the constant builds and types {@code consume}'s
   * parameter to complex64. The shape is unknown (⊤) rather than {@code (2,)}: skipping the fold
   * leaves the list elements non-constant, so the size isn't recovered (the integer-valued {@link
   * #testConstantComplex64()} still gets {@code (2,)}). TODO: recover the shape, tracked by <a
   * href="https://github.com/wala/ML/issues/644">wala/ML#644</a>.
   */
  @Test
  public void testComplexLiteralEntrypoint()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_complex_literal.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(new TensorType(COMPLEX64, null))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/640">wala/ML#640</a>: the
   * constant folder evaluates a foldable expression (in an uncalled function) that raises at
   * evaluation time -- here {@code 1 / 0} ({@code ZeroDivisionError}); the original NLPGNN case was
   * a {@code NameError} on a free name. Folding must skip such an eval-time error rather than abort
   * the class hierarchy. If the hierarchy builds, {@code consume}'s parameter types normally to
   * {@code (3,)} int32.
   */
  @Test
  public void testFoldingEvalError()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_folding_eval_error.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>: a tensor
   * passed to a Keras {@code call} method types its parameter. {@code BiLSTM.call}'s {@code inputs}
   * receives a token-id tensor (which then feeds an {@code Embedding}), so it types to {@code (1,
   * 3)} int32 rather than being missed.
   */
  @Test
  public void testKerasCallEmbeddingParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_keras_call_embedding_param.py",
        "BiLSTM.call",
        1,
        2,
        Map.of(3, Set.of(TensorType.of(INT_32, 1, 3))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>: a {@code
   * tf.data} dataset element passed to a function types its parameter. {@code target_convert}'s
   * {@code targets} receives a dataset element, so it types to {@code (2,)} int32 rather than being
   * missed.
   */
  @Test
  public void testDatasetElementParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset_element_param.py",
        "target_convert",
        1,
        2,
        Map.of(2, Set.of(TensorType.of(INT_32, 2))));
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
  public void testRaggedFromRowStartsFull()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_starts_full.py",
        "test_ragged_from_row_starts_full",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_5_RAGGED_INT32),
            3, Set.of(TENSOR_5_RAGGED_INT32),
            4, Set.of(TENSOR_5_RAGGED_INT32),
            5, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  /**
   * Coverage guard for the {@code DynamicDim} fallback branch in {@link
   * com.ibm.wala.cast.python.ml.client.RaggedFromRowStarts}'s nrows inference (<a
   * href="https://github.com/wala/ML/issues/545">wala/ML#545</a>). When {@code row_starts} has a
   * non-{@code NumericDim} first dim — here, a {@code DynamicDim} from {@code tf.keras.Input}'s
   * symbolic batch axis — the generator emits {@code DynamicDim.INSTANCE} for the inferred nrows,
   * yielding a {@code (DynamicDim, RaggedDim)} shape rather than emitting raw {@code null}.
   */
  @Test
  public void testRaggedFromRowStartsDynamicRowDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_starts_dynamic.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_NONE_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromRowLengths()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_lengths.py",
        "test_ragged_from_row_lengths",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromRowLimits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_limits.py",
        "test_ragged_from_row_limits",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromValueRowIds()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_value_rowids.py",
        "test_ragged_from_value_rowids",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_4_RAGGED_INT32),
            3, Set.of(TENSOR_4_RAGGED_INT32),
            4, Set.of(TENSOR_4_RAGGED_INT32)));
  }

  /**
   * Coverage guard for the {@code DynamicDim} fallback branch in {@link
   * com.ibm.wala.cast.python.ml.client.RaggedFromRowLengths}'s nrows inference (<a
   * href="https://github.com/wala/ML/issues/545">wala/ML#545</a>).
   */
  @Test
  public void testRaggedFromRowLengthsDynamicRowDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_lengths_dynamic.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_NONE_RAGGED_INT32)));
  }

  /**
   * Coverage guard for the {@code DynamicDim} fallback branch in {@link
   * com.ibm.wala.cast.python.ml.client.RaggedFromRowLimits}'s nrows inference (<a
   * href="https://github.com/wala/ML/issues/545">wala/ML#545</a>).
   */
  @Test
  public void testRaggedFromRowLimitsDynamicRowDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_limits_dynamic.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_NONE_RAGGED_INT32)));
  }

  /**
   * Coverage guard for the {@code DynamicDim} fallback branch in {@link
   * com.ibm.wala.cast.python.ml.client.RaggedFromRowSplits}'s nrows inference (<a
   * href="https://github.com/wala/ML/issues/545">wala/ML#545</a>).
   */
  @Test
  public void testRaggedFromRowSplitsDynamicRowDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_splits_dynamic.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_NONE_RAGGED_INT32)));
  }

  @Test
  public void testStrictnessFailure()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_strictness_failure.py",
        "test_strictness",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_INT32)));
  }

  @Test
  public void testNoneDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_none_dtype.py", "test_none_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testRaggedKeywordArgsNested()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_keyword_args_nested.py",
        "test_keywords",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32),
            3, Set.of(TENSOR_2_RAGGED_RAGGED_INT32),
            4, Set.of(TENSOR_2_RAGGED_RAGGED_INT32),
            5, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedKeywordArgsAdditional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_keyword_args_more.py",
        "test_keywords",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_5_RAGGED_INT32),
            3, Set.of(TENSOR_4_RAGGED_INT32),
            4, Set.of(TENSOR_4_RAGGED_INT32),
            5, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testRaggedKeywordArgs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_keyword_args.py",
        "test_keywords",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_2_RAGGED_INT32),
            3, Set.of(TENSOR_2_RAGGED_INT32),
            4, Set.of(TENSOR_2_RAGGED_INT32),
            5, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedKeywordArgsV2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_keyword_args_v2.py",
        "test_keywords",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_2_RAGGED_INT32),
            3, Set.of(TENSOR_2_RAGGED_INT32),
            4, Set.of(TENSOR_2_RAGGED_INT32),
            5, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedKeywordArgsMixed()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ragged_mixed_new.py",
        "test_ragged_mixed_args_new",
        3,
        3,
        Map.of(
            // rt1: positional values, keyword value_rowids. inferred nrows=3.
            2,
            Set.of(TENSOR_3_RAGGED_INT32),

            // rt2: positional values, keyword value_rowids, keyword nrows=5.
            3,
            Set.of(TENSOR_5_RAGGED_INT32),

            // rt3: positional values, positional value_rowids, keyword nrows=3.
            4,
            Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedNrowsPositional()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ragged_nrows_positional.py",
        "test_ragged_nrows_positional",
        1,
        1,
        Map.of(
            // rt: positional values, positional value_rowids, positional nrows=3.
            2, Set.of(TENSOR_3_RAGGED_INT32)));
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

  @Test
  public void testSparseAdd()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseAdd2()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseAdd3()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32.asSparse())));
  }

  @Test
  public void testSparseAdd4()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32.asSparse())));
  }

  @Test
  public void testSparseAdd5()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseAdd6()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32.asSparse())));
  }

  /**
   * {@code run_optimization(x, y)} receives batched CIFAR-10 data (not MNIST despite the file's
   * docstring). At runtime {@code x} has shape {@code (4096, 32, 32, 3)} dtype {@code float32} and
   * {@code y} has shape {@code (4096,)} dtype {@code uint8} (verified by Python assert statements
   * in {@code multigpu_training.py}).
   *
   * <p>Master's types for values 2 and 3 are {@code MNIST_INPUT} &mdash; the MNIST shape {@code (n,
   * 28, 28)} &mdash; which is <em>confidently wrong</em> for CIFAR-10 data (wala/ML#393: {@code
   * keras.datasets.X.load_data()} is seeded uniformly as MNIST-shaped regardless of which dataset
   * module {@code X} is). Branch now hardcodes {@code tf.keras.datasets.cifar10.load_data()} as an
   * intrinsic (paralleling the MNIST modeling), so value 2 correctly reports {@code (4096, 32, 32,
   * 3) float32}. Value 3 (labels) now correctly reports {@code (4096,) uint8} after wala/ML#410
   * landed the top-level {@code np.reshape(x, shape)} modeling: {@code y_train =
   * np.reshape(y_train, (-1))} at line 66 of the source is the function form of reshape (as opposed
   * to the method form {@code x.reshape(...)} which was already handled by {@link NdarrayReshape}).
   * The fix added a {@code numpy/reshape} class to {@code numpy.xml} paired with an {@link
   * NpReshape} generator that reuses {@link Reshape}'s {@code -1}-inference logic and also accepts
   * a bare integer as the shape argument (the parenthesised {@code (-1)} parses as {@code -1}, not
   * a 1-tuple).
   *
   * <p>Expected tensor variable count: 5. The historical trajectory is: pre-wala/ML#451 the count
   * was 5 because of a spurious classification at {@code vn=44} ({@code gpu_batch_size =
   * int(batch_size / num_gpus)} at line 222) where {@code batch_size / num_gpus} is a binop on
   * Python ints, dispatched to {@link ElementWiseOperation} and typed {@code [] of int32} via
   * {@link TensorGenerator#getDTypesOfValue}'s Integer-constant → INT32 path. wala/ML#451's binop
   * operand-tensor gate rejected that classification, dropping the count to the master baseline of
   * 4. wala/ML#430's {@code Gradient} generator then added one fresh tensor per {@code
   * tape.gradient(...)} call (one such call here), bringing the count back to 5 — but now backed by
   * the legitimate registration of the gradient result rather than the spurious binop pickup. See
   * also wala/ML#361 (MNIST modeling) and wala/ML#393 (CIFAR-10 modeling, closed by the commit that
   * landed this test's partial pass).
   */
  @Test
  public void testMultiGPUTraining()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "multigpu_training.py",
        "run_optimization",
        2,
        5,
        Map.of(2, Set.of(TENSOR_4096_32_32_3_FLOAT32), 3, Set.of(TENSOR_4096_UINT8)));
  }

  /**
   * Companion to {@link #testMultiGPUTraining()} on the {@code average_gradients} function in the
   * same fixture. Exercises tensor classification through a per-tower gradient-averaging loop: for
   * each gradient {@code g} in the input tower-gradient list, the body computes {@code
   * tf.expand_dims(g, 0)}, collects the results into a list, and reduces them with {@code
   * tf.concat(axis=0, values=grads)}. Verifies the analyzer detects the 3 internal tensor variables
   * this loop produces: {@code tf.expand_dims(g, 0)}'s dedicated {@code <new>} allocation, the
   * post-allocation receiver, and {@code tf.concat(...)} flowing through <a
   * href="https://github.com/wala/ML/issues/196">wala/ML#196</a>'s {@link
   * com.ibm.wala.cast.python.ml.client.ReadDataFallback}.
   *
   * <p>With `list.append` modeled (<a
   * href="https://github.com/wala/ML/issues/136">wala/ML#136</a>), the loop's element values reach
   * classification and the local-tensor count is 8 (the loop variable {@code g}, the per-iteration
   * {@code expand_dims} results, and the {@code concat} chain). The parameter count stays 0: {@code
   * tower_grads} is the list <em>container</em>, which is not itself a tensor; its elements are
   * classified where they are read.
   */
  @Test
  public void testMultiGPUTraining2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("multigpu_training.py", "average_gradients", 0, 8);
  }

  @Test
  public void testReduceMean()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  // wala/ML#449 Tier 3: reductions. Each collapses dims along `axis` (default `None` = all dims)
  // and preserves dtype from input (except `reduce_all` which is always BOOL).

  @Test
  public void testReduceMax()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_max.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.reduce_min(input_tensor)}. Mirrors {@link
   * #testReduceMax()} — the {@link com.ibm.wala.cast.python.ml.client.ReduceMin} generator extends
   * {@link com.ibm.wala.cast.python.ml.client.ReduceMax}, sharing the axis-collapse / keepdims
   * shape inference and input-dtype passthrough.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testReduceMin()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_min.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testReduceProd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_prod.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testReduceLogSumExp()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_logsumexp.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testReduceAll()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_all.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_BOOL)));
  }

  /**
   * Verifies that {@code tf.estimator.EstimatorSpec(...)} produces a fresh allocation with each
   * named parameter stored as a field on the result. The test reads {@code spec.loss} and asserts
   * that it round-trips back to the original {@code loss_tensor} (a scalar float32). If
   * EstimatorSpec is mis-modeled as "return one of the inputs" instead of "fresh allocation with
   * field sets," the read would either return the wrong dtype or fail to resolve. Exercises the
   * binding + body fix from wala/ML#429.
   */
  @Test
  public void testEstimatorSpec()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_estimator_spec.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Regression test for <a href="https://github.com/wala/ML/issues/523">wala/ML#523</a>: the {@code
   * EstimatorSpec.do()} constructor's fresh allocation must be a {@code
   * Ltensorflow/estimator/EstimatorSpec} (a namedtuple-like spec object) rather than a {@code
   * Ltensorflow/python/framework/ops/Tensor}. The fixture passes the {@code spec} directly to
   * {@code f}; if {@code spec} is misclassified as a tensor allocation, {@code f}'s parameter would
   * be a tensor (and the expected count would be 1 / 1). With the correct non-tensor class, {@code
   * f} has 0 tensor parameters and 0 tensor variables.
   */
  @Test
  public void testEstimatorSpecNotTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_estimator_spec_not_tensor.py", "f", 0, 0, Map.of());
  }

  @Test
  public void testReduceMean2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceMean3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean.py", "h", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceMean4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceMean5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceMean6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "h", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testReduceMean7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "i", 1, 1, Map.of(2, Set.of(TENSOR_2_1_FLOAT32)));
  }

  @Test
  public void testReduceMean8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "j", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testReduceSum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Guards that {@code tf.reduce_sum} preserves an integer input's dtype rather than promoting it
   * to {@code float32}. Unlike {@code tf.reduce_mean}, summing integers yields an integer;
   * previously the {@code Reduce*} ops inherited {@code ReduceMean}'s mean-specific promotion via
   * {@code extends ReduceMean}, so {@code reduce_sum} of an {@code int32} was mistyped {@code
   * float32}. Fixed by the {@code Reduction}-base hierarchy (wala/ML#514).
   */
  @Test
  public void testReduceSumInt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum_int.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  // wala/ML#449 Tier 4: intrinsic-API ops with fixed output shape and dtype. Both `tf.rank` and
  // `tf.size` return scalar int32 regardless of input. Same hardcoded-output shape as
  // `DatasetRangeGenerator`'s `[] of int64`.

  @Test
  public void testRank()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_rank.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testSize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_size.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  // wala/ML#449 Tier 6: argmax/argmin produce int64 indices. Output shape is the input shape with
  // the `axis` dimension removed (via `ReduceMean`), unblocked by the per-context layer-output
  // fix (wala/ML#530) that stopped `testNeuralNetwork*` from regressing on the
  // `ElementWiseOperation`
  // cross-context cartesian pair.

  /**
   * Verifies that {@code tf.math.argmax(x, axis=0)} routes through the dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Argmax} generator and emits the precise {@code int64} dtype
   * (the TF default for argmax indices) and the precise {@code (3,)} shape ({@code (2, 3)} with
   * {@code axis=0} removed). Shape precision was unblocked by wala/ML#530.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgmax()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_argmax.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT64)));
  }

  /**
   * Counterpart of {@link #testArgmax()} for {@code tf.math.argmin}. Same semantics: dtype defaults
   * to {@code int64} (overridable via {@code output_type}, see {@link #testArgminOutputType()}),
   * and shape is the precise {@code (3,)} ({@code (2, 3)} with {@code axis=0} removed), unblocked
   * by wala/ML#530. See {@link com.ibm.wala.cast.python.ml.client.Argmin}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgmin()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_argmin.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT64)));
  }

  /**
   * Verifies that {@code tf.math.argmax(x, axis=0, output_type=tf.int32)} honors the explicit
   * {@code output_type} override and emits an {@code int32}-dtype tensor instead of the {@code
   * int64} default. Exercises the dtype-arg dispatch path on {@link
   * com.ibm.wala.cast.python.ml.client.Argmax} after the wala/ML#463 fix. The fixture's sink {@code
   * f(x, y)} has two parameters so that each tensor's inferred type can be checked independently.
   * The result {@code y} (vn=3) has the precise {@code (3,) int32} shape ({@code (2, 3)} with
   * {@code axis=0} removed), unblocked by wala/ML#530.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgmaxOutputType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_argmax_output_type.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_3_FLOAT32),
            3, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Like {@link #testArgmaxOutputType()} but passes {@code output_type} positionally ({@code
   * tf.math.argmax(x, 0, tf.int32)}). Shape inference must not misread the positional {@code
   * output_type} argument as {@link com.ibm.wala.cast.python.ml.client.ReduceMean}'s {@code
   * keepdims} (they share positional index 2); the result {@code y} (vn=3) is the precise {@code
   * (3,) int32}, not a {@code keepdims=true} union (e.g. {@code (1, 3)}). Regression guard for the
   * {@code getKeepDimsValues} override on {@link com.ibm.wala.cast.python.ml.client.Argmax}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgmaxOutputTypePositional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_argmax_output_type_positional.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_3_FLOAT32),
            3, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Tests that {@code tf.math.argmax} resolves its input when passed by keyword ({@code
   * tf.math.argmax(input=x, axis=0)}). {@link com.ibm.wala.cast.python.ml.client.Argmax} delegates
   * shape inference to {@link com.ibm.wala.cast.python.ml.client.ReduceMean}, whose input parameter
   * is named {@code input_tensor}; argmax's is named {@code input}, so without overriding the
   * input-parameter name the keyword lookup fails and shape inference throws {@code
   * IllegalStateException}. The result {@code y} (vn=3) is the precise {@code (3,) int64}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgmaxInputKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_argmax_input_keyword.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_3_FLOAT32),
            3, Set.of(TENSOR_3_INT64)));
  }

  /**
   * Counterpart of {@link #testArgmaxOutputType()} for {@code tf.math.argmin}. Same dispatch path
   * via the inherited {@link com.ibm.wala.cast.python.ml.client.Argmin} extends {@link
   * com.ibm.wala.cast.python.ml.client.Argmax} relationship; the result {@code y} (vn=3) has the
   * precise {@code (3,) int32} shape, unblocked by wala/ML#530.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgminOutputType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_argmin_output_type.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_3_FLOAT32),
            3, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Companion to {@link #testArgmaxOutputType()} that exercises the *single-parameter sink, two
   * call sites* shape: {@code def f(a): ...; f(x); f(y)}. Parameter {@code a} should union {@code
   * x}'s and {@code y}'s tensor types across the two call sites &mdash; verifies that the {@code
   * output_type=tf.int32} override on {@code y} survives the second sink call rather than being
   * clobbered by the {@code int64} default. The {@code y} contribution to the union is the precise
   * {@code (3,) int32} shape, unblocked by wala/ML#530.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgmaxOutputTypeDoubleSink()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_argmax_output_type_double_sink.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_3_INT32)));
  }

  /**
   * Counterpart of {@link #testArgmaxOutputTypeDoubleSink()} for {@code tf.math.argmin}; the {@code
   * y} contribution to the union is the precise {@code (3,) int32} shape, unblocked by wala/ML#530.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgminOutputTypeDoubleSink()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_argmin_output_type_double_sink.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_3_INT32)));
  }

  // wala/ML#449 Tier 7: linspace/broadcast_to. Shape derives from a shape-arg (`num`/`shape`),
  // dtype derives from a value-arg (`start`/`input`).

  /**
   * Verifies that {@code tf.linspace(0.0, 10.0, 5)} routes through the dedicated {@link Linspace}
   * generator and emits the precise rank-1 shape {@code (5,)} with {@code float32} dtype (derived
   * from the float-typed {@code start} argument).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLinspace()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_linspace.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  /**
   * Companion to {@link #testLinspace()} that exercises the integer-promotion branch in {@link
   * com.ibm.wala.cast.python.ml.client.Linspace#getDefaultDTypes}. {@code tf.linspace} with integer
   * {@code start}/{@code stop} promotes the output to {@code float64} (verified on TF 2.9), not
   * {@code float32}. The float-input case is covered by {@link #testLinspace()}.
   */
  @Test
  public void testLinspaceInt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_linspace_int.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT64)));
  }

  /**
   * Drives the {@code axis}-passed branch in {@link
   * com.ibm.wala.cast.python.ml.client.Linspace#getDefaultShapes}. With {@code axis=1} and vector
   * {@code start}/{@code stop}, {@code tf.linspace} interpolates along axis 1, producing a
   * higher-rank result whose runtime shape is {@code (2, 5)} with dtype {@code float32}.
   *
   * <p>The static analysis currently returns ⊤ (unknown shape) for any axis-passed call — combining
   * {@code start}'s rank with {@code num} is not yet implemented; the generator trades precision
   * for soundness. The assertion encodes the observed result with a TODO pointing at the precision
   * improvement that would narrow it.
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/475">wala/ML#475</a> is fixed and
   * {@code Linspace.getDefaultShapes} computes the precise output shape from {@code
   * start.shape[:axis] + (num,) + start.shape[axis:]}, narrow the assertion to {@code
   * Set.of(TENSOR_2_5_FLOAT32)} (a new constant).
   *
   * <p>Companion to {@link #testLinspace()} (covering the absent-axis branch).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLinspaceAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_linspace_axis.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.broadcast_to(x, [2, 3])} routes through the dedicated {@link
   * BroadcastTo} generator and emits shape {@code (2, 3)} (read from the {@code shape} argument's
   * literal list) with {@code float32} dtype (derived from the {@code input} tensor).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBroadcastTo()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_broadcast_to.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Regression guard for {@code tf.broadcast_to(x, tf.shape(y))} &mdash; the runtime-tensor
   * shape-arg pattern. {@link com.ibm.wala.cast.python.ml.client.TensorGenerator
   * #getShapesFromShapeArgument} throws {@link IllegalStateException} for the runtime {@code
   * Ltensorflow/python/framework/ops/Tensor} that {@code tf.shape(y)} now allocates (post the
   * wala/ML#489 root-cause fix on this PR's `tensorflow.xml`); {@link
   * com.ibm.wala.cast.python.ml.client.BroadcastTo#getDefaultShapes}'s try/catch returns {@code
   * null} (lattice ⊤) instead of letting the exception abort the analysis. The result is shape ⊤
   * with dtype inherited from {@code x} (float32). Without the catch, analysis aborts and this test
   * fails &mdash; this is the direct regression guard for this PR's localized-tolerance fix.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/473">wala/ML#473</a>): the runtime answer is
   * (2, 3) of float32. When the helper learns to recognize {@code tf.shape(y)} as a shape arg
   * (rather than treating it as an unmodeled runtime tensor), tighten this assertion from {@link
   * #TENSOR_UNKNOWN_SHAPE_FLOAT32} to {@code TENSOR_2_3_FLOAT32}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBroadcastToRuntimeShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_broadcast_to_runtime_shape.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.linalg.tensordot} with a scalar {@code axes}. Output
   * dtype is inherited from the {@code a} input (here float32); with {@code axes=1} the output
   * shape is {@code a.shape[:-1] + b.shape[1:]}, so two (2, 2) inputs yield (2, 2). See {@link
   * com.ibm.wala.cast.python.ml.client.Tensordot} (wala/ML#449).
   */
  @Test
  public void testTensordot()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensordot.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.linalg.trace}. Output dtype is inherited from the {@code
   * x} input (here float32); the output shape is the input shape with the last two dimensions
   * dropped, so a (2, 2) input yields a scalar. See {@link
   * com.ibm.wala.cast.python.ml.client.Trace} (wala/ML#449).
   */
  @Test
  public void testTrace()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_trace.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.linalg.trace} on a batched input. The trace collapses the
   * last two dimensions, so a (3, 2, 2) input yields a (3,) output that inherits the input dtype.
   * Exercises the leading-dimensions path of {@link com.ibm.wala.cast.python.ml.client.Trace}
   * (wala/ML#449).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTraceBatched()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_trace_batched.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_3_2_2_FLOAT32),
            3, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.transpose(a)} with no {@code perm} reverses the axes, so a {@code (2,
   * 3)} input yields {@code (3, 2)}. Previously modeled as a first-argument {@code pass_through},
   * which reported the input shape unchanged. See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTransposeDefault()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_transpose.py",
        "consume_transpose_default",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.transpose(a, perm)} with a constant {@code perm} permutes the axes so
   * output axis {@code i} is input axis {@code perm[i]}: a {@code (2, 3, 4)} input with {@code perm
   * = [0, 2, 1]} yields {@code (2, 4, 3)}. Previously modeled as a first-argument {@code
   * pass_through}, which reported the input shape unchanged. See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTransposePerm()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_transpose.py",
        "consume_transpose_perm",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_4_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.transpose} resolves a {@code perm} passed as a tensor constant (rather
   * than a Python list literal): a {@code (2, 3, 4)} input with {@code perm = tf.constant([2, 1,
   * 0])} permutes precisely to {@code (4, 3, 2)}. Exercises the tensor-constant {@code perm}
   * resolution path of {@link com.ibm.wala.cast.python.ml.client.Transpose}, distinct from the
   * list-literal path. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket
   * 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTransposeTensorPerm()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_transpose.py",
        "consume_transpose_tensor_perm",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_3_2_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.diag(diagonal)} increases rank by one: it places the input's
   * last axis on the diagonal of a new trailing square, so a {@code (4,)} input yields {@code (4,
   * 4)}. Previously modeled as a first-argument {@code pass_through}, which reported the input
   * shape unchanged. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDiag()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_linalg_rank_delta.py",
        "consume_diag",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.diag_part(input)} decreases rank by one: it extracts the
   * diagonal of each trailing square, so a {@code (3, 3)} input yields {@code (3,)}. Previously
   * modeled as a first-argument {@code pass_through}, which reported the input shape unchanged. See
   * <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDiagPart()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_linalg_rank_delta.py",
        "consume_diag_part",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.matrix_transpose(a)} swaps the last two axes, preserving leading
   * batch dimensions, so a {@code (2, 3)} input yields {@code (3, 2)}. Previously modeled as a
   * first-argument {@code pass_through}, which reported the input shape unchanged. See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testMatrixTranspose()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_linalg_rank_delta.py",
        "consume_matrix_transpose",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.adjoint(matrix)} (conjugate transpose) swaps the last two axes
   * exactly like {@code matrix_transpose}, so a {@code (2, 3)} input yields {@code (3, 2)}.
   * Previously modeled as a first-argument {@code pass_through}, which reported the input shape
   * unchanged. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testAdjoint()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_linalg_rank_delta.py",
        "consume_adjoint",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.tile(input, multiples)} multiplies each axis by the corresponding entry
   * of {@code multiples}, so a {@code (2, 3)} input tiled by {@code [2, 1]} yields {@code (4, 3)}.
   * Previously modeled as a first-argument {@code pass_through}, which reported the input shape
   * unchanged. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTile()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tile.py", "consume_tile", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.solve(matrix, rhs)} reports the shape and dtype of the
   * right-hand side {@code rhs} (arg 1), not the coefficient {@code matrix} (arg 0). A {@code (3,
   * 3)} matrix and a {@code (3, 5)} rhs yield a {@code (3, 5)} result. Previously the op was
   * modeled as a first-argument {@code pass_through}, which reported the {@code (3, 3)} matrix
   * shape — an actively wrong answer. See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2b.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSolve()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_solve.py", "consume_solve", 1, 1, Map.of(2, Set.of(TENSOR_3_5_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.cholesky_solve(chol, rhs)} reports the shape and dtype of the
   * right-hand side {@code rhs} (arg 1), not the Cholesky factor {@code chol} (arg 0). See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2b.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCholeskySolve()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_solve.py", "consume_cholesky_solve", 1, 1, Map.of(2, Set.of(TENSOR_3_5_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linalg.triangular_solve(matrix, rhs)} reports the shape and dtype of
   * the right-hand side {@code rhs} (arg 1), not the coefficient {@code matrix} (arg 0). See <a
   * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2b.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTriangularSolve()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_solve.py",
        "consume_triangular_solve",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_5_FLOAT32)));
  }

  /**
   * Tier-6 op (wala/ML#449): {@code tf.sort(values, ...)}. The XML routes the call through {@code
   * convert_to_tensor} of {@code values}, so shape and dtype pass through unchanged — no dedicated
   * generator needed.
   */
  @Test
  public void testSort()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sort.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_6_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.tensor_scatter_nd_update}. Output dtype AND shape are
   * inherited from the {@code tensor} input — true shape-and-dtype passthrough on the first arg.
   * Here the input is shape {@code (4,)} float32, so the precise expected result is {@code (4,)
   * float32}. See {@link com.ibm.wala.cast.python.ml.client.TensorScatterNdUpdate} (wala/ML#449).
   *
   * <p>Post-master-merge of wala/ML#380's `tensor_scatter_nd_update` inlining (#237), the analysis
   * also produces an additional fully-⊤ context — likely a context where the input arg's PTS isn't
   * recovered through the inlined synthetic body, so the passthrough returns ⊤/UNKNOWN. The
   * assertion captures both contexts (per the prefer-observed-assertion convention from
   * `CONTRIBUTING.md`); a precision improvement that eliminates the ⊤ context will narrow the
   * actual to just {@code TENSOR_4_FLOAT32} and this test will fail with a clear "expected union,
   * got per-context" diff — that's the cue to update.
   *
   * <p>TODO(wala/ML#474): Once the additional fully-⊤ context for the inlined-passthrough path is
   * investigated/eliminated, narrow the assertion to {@code Set.of(TENSOR_4_FLOAT32)}.
   */
  @Test
  public void testTensorScatterNdUpdate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tensor_scatter_nd_update.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_FLOAT32, TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Generator-dispatch test for {@code tf.sequence_mask}. With no {@code dtype} argument the output
   * dtype is the TF-default {@code bool}; with a constant {@code maxlen} the shape is {@code
   * lengths.shape + [maxlen]}, so {@code sequence_mask([1, 3, 2], maxlen=5)} yields (3, 5).
   * (wala/ML#449 Tier 8.)
   */
  @Test
  public void testSequenceMask()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sequence_mask.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_5_BOOL)));
  }

  /**
   * Generator-dispatch test for {@code tf.sequence_mask} with an explicit {@code dtype} override
   * ({@code tf.sequence_mask(..., maxlen=5, dtype=tf.int32)}). The output dtype follows the
   * argument ({@code int32}) rather than the default {@code bool}, and the constant {@code maxlen}
   * gives the precise (3, 5) shape. Regression guard for surfacing the {@code dtype} parameter
   * through {@link com.ibm.wala.cast.python.ml.client.SequenceMask}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSequenceMaskDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sequence_mask_dtype.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_5_INT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.nn.embedding_lookup}. Output dtype is inherited from the
   * {@code params} input (here float32); the output shape is {@code ids.shape + params.shape[1:]},
   * so a (3, 2) table looked up by (2,) ids yields (2, 2). See {@link
   * com.ibm.wala.cast.python.ml.client.EmbeddingLookup} (wala/ML#449 Tier 8).
   */
  @Test
  public void testEmbeddingLookup()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_embedding_lookup.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.gather_nd}. Output dtype is inherited from the {@code
   * params} input (here float32); the output shape is {@code indices.shape[:-1] +
   * params.shape[indices.shape[-1]:]}, so a (2, 2) table indexed by (2, 2) depth-2 indices yields
   * (2,). See {@link com.ibm.wala.cast.python.ml.client.GatherNd} (wala/ML#449 Tier 8).
   */
  @Test
  public void testGatherNd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gather_nd.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.boolean_mask}. Output dtype is inherited from the {@code
   * tensor} input (here float32); the masked axis collapses to a dynamic dimension (the runtime
   * {@code True} count), so masking a (3, 2) tensor with a rank-1 mask yields {@code [Dynamic, 2]}.
   * The leading dimension is inherently runtime, so it stays dynamic. See {@link
   * com.ibm.wala.cast.python.ml.client.BooleanMask} (wala/ML#449 Tier 8).
   */
  @Test
  public void testBooleanMask()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_boolean_mask.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.image.extract_patches}. Output dtype is inherited from
   * the {@code images} input (here float32); the output shape is {@code [batch, out_rows, out_cols,
   * sizes_r * sizes_c * channels]}, so a (1, 10, 10, 3) image with {@code sizes=[1, 3, 3, 1]},
   * {@code strides=[1, 5, 5, 1]}, {@code rates=[1, 1, 1, 1]} and {@code VALID} padding yields (1,
   * 2, 2, 27). See {@link com.ibm.wala.cast.python.ml.client.ExtractPatches} (wala/ML#449 Tier 8).
   */
  @Test
  public void testExtractPatches()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_extract_patches.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_2_2_27_FLOAT32)));
  }

  /**
   * Regression guard for {@code tf.image.extract_patches} called with a Python <em>list
   * literal</em> {@code images} argument (rather than a {@code tf.Tensor}), per <a
   * href="https://github.com/wala/ML/issues/584">wala/ML#584</a>. The list-literal shape {@code (1,
   * 1, 1, 1)} is recovered from the nesting structure, and the result is the concrete {@code (1, 0,
   * 0, 9) int32}: a {@code 3x3} patch does not fit the {@code 1x1} image, so {@code VALID} padding
   * yields a 0-extent spatial output (depth {@code 3*3*1 = 9}), matching the runtime shape the
   * Python fixture asserts (<a href="https://github.com/wala/ML/issues/585">wala/ML#585</a>).
   */
  @Test
  public void testExtractPatches2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_extract_patches2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_0_0_9_INT32)));
  }

  /**
   * Regression guard for {@code tf.image.extract_patches} called with an {@code images} argument
   * built from a nested list <em>comprehension</em> (the wala/ML#584 corpus case), per <a
   * href="https://github.com/wala/ML/issues/584">wala/ML#584</a>. Resolving such an {@code images}
   * operand throws inside the generator; before the fix that aborted the whole type computation and
   * the result dropped its tensor classification entirely. The result must still be recognized as a
   * tensor — here ⊤ shape and ⊤ dtype, since a comprehension's computed elements (unlike a
   * literal's constants) yield no statically inferable dtype.
   */
  @Test
  public void testExtractPatchesComprehension()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_extract_patches_comprehension.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /** Pure-passthrough generator test for {@code tf.math.tan} (wala/ML#422). */
  @Test
  public void testTan()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tan.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.asin} (wala/ML#422). */
  @Test
  public void testAsin()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_asin.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.atan} (wala/ML#422). */
  @Test
  public void testAtan()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_atan.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.sinh} (wala/ML#422). */
  @Test
  public void testSinh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sinh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.cosh} (wala/ML#422). */
  @Test
  public void testCosh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_cosh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.asinh} (wala/ML#422). */
  @Test
  public void testAsinh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_asinh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.acosh} (wala/ML#422). */
  @Test
  public void testAcosh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_acosh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.atanh} (wala/ML#422). */
  @Test
  public void testAtanh()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_atanh.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.log1p} (wala/ML#422). */
  @Test
  public void testLog1p()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_log1p.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.expm1} (wala/ML#422). */
  @Test
  public void testExpm1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_expm1.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.round} (wala/ML#422). */
  @Test
  public void testRound()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_round.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.reciprocal} (wala/ML#422). */
  @Test
  public void testReciprocal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reciprocal.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.softplus} (wala/ML#422). */
  @Test
  public void testSoftplus()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_softplus.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.softsign} (wala/ML#422). */
  @Test
  public void testSoftsign()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_softsign.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.square} (wala/ML#422). */
  @Test
  public void testSquare()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_square.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.erf} (wala/ML#422). */
  @Test
  public void testErf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_erf.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Pure-passthrough generator test for {@code tf.math.erfc} (wala/ML#422). */
  @Test
  public void testErfc()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_erfc.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Elementwise binary generator test for {@code tf.math.atan2} (wala/ML#422). */
  @Test
  public void testAtan2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_atan2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Elementwise binary generator test for {@code tf.math.maximum} (wala/ML#422). */
  @Test
  public void testMaximum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_maximum.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Elementwise binary generator test for {@code tf.math.minimum} (wala/ML#422). */
  @Test
  public void testMinimum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_minimum.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testAtan2}: {@code tf.math.atan2(y=..., x=...)}. Exercises
   * the kw-arg-resolution path on the {@code ElementWiseOperation} dispatch.
   */
  @Test
  public void testAtan2Kw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_atan2_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testMaximum}: {@code tf.math.maximum(x=..., y=...)}.
   * Exercises the kw-arg-resolution path on the {@code ElementWiseOperation} dispatch.
   */
  @Test
  public void testMaximumKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_maximum_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testMinimum}: {@code tf.math.minimum(x=..., y=...)}.
   * Exercises the kw-arg-resolution path on the {@code ElementWiseOperation} dispatch.
   */
  @Test
  public void testMinimumKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_minimum_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.fashion_mnist.load_data()}. Shapes and dtype are
   * identical to {@code mnist.load_data()}. The fixture passes all four unpacked arrays ({@code
   * x_train}, {@code y_train}, {@code x_test}, {@code y_test}) into the 4-arg sink, so the
   * assertion pins types at {@code vn=2..5}.
   */
  @Test
  public void testFashionMnistLoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_fashion_mnist_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_60000_28_28_UINT8),
            3, Set.of(TENSOR_60000_UINT8),
            4, Set.of(TENSOR_10000_28_28_UINT8),
            5, Set.of(TENSOR_10000_UINT8)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.cifar100.load_data()}. Shapes are identical to
   * {@code cifar10.load_data()}, but the {@code y_train} / {@code y_test} dtype is {@code int64}
   * (cifar100's class indices) rather than {@code uint8} (cifar10's class indices). The dispatch
   * routes through the dedicated {@link com.ibm.wala.cast.python.ml.client.Cifar100InputData}
   * generator (closes wala/ML#487's mis-routing through {@code Cifar10InputData}). Asserts on all
   * four unpacked arrays at {@code vn=2..5}.
   */
  @Test
  public void testCifar100LoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_cifar100_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_50000_32_32_3_UINT8),
            3, Set.of(TENSOR_50000_1_INT64),
            4, Set.of(TENSOR_10000_32_32_3_UINT8),
            5, Set.of(TENSOR_10000_1_INT64)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.reuters.load_data()}. Asserts on all four unpacked
   * arrays at {@code vn=2..5}: {@code x_train} ({@code (8982,)} {@code object} &mdash; newswires
   * are variable-length integer-encoded sequences, so numpy stores them in an {@code object}
   * array), {@code y_train} ({@code (8982,)} {@code int64}), {@code x_test} ({@code (2246,)} {@code
   * object}), {@code y_test} ({@code (2246,)} {@code int64}). The {@code object} dtype matches the
   * runtime truth the Python fixture asserts (wala/ML#488).
   */
  @Test
  public void testReutersLoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reuters_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_8982_OBJECT),
            3, Set.of(TENSOR_8982_INT64),
            4, Set.of(TENSOR_2246_OBJECT),
            5, Set.of(TENSOR_2246_INT64)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.boston_housing.load_data()}. Asserts on all four
   * unpacked arrays at {@code vn=2..5}: {@code x_train} ({@code (404, 13)} {@code float64}
   * features), {@code y_train} ({@code (404,)} {@code float64} regression targets), {@code x_test}
   * ({@code (102, 13)} {@code float64}), {@code y_test} ({@code (102,)} {@code float64}).
   */
  @Test
  public void testBostonHousingLoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_boston_housing_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_404_13_FLOAT64),
            3, Set.of(TENSOR_404_FLOAT64),
            4, Set.of(TENSOR_102_13_FLOAT64),
            5, Set.of(TENSOR_102_FLOAT64)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.imdb.load_data()}. The four returned arrays each
   * have shape {@code (25000,)}: the {@code x_train} / {@code x_test} arrays carry numpy {@code
   * object} dtype (variable-length integer-encoded sequences, so numpy stores them in an {@code
   * object} array); the {@code y_train} / {@code y_test} arrays have dtype {@code int64} (binary
   * labels). Asserts on all four unpacked arrays at {@code vn=2..5}. The {@code object} dtype
   * matches the runtime truth the Python fixture asserts (wala/ML#488).
   */
  @Test
  public void testImdbLoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_imdb_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_25000_OBJECT),
            3, Set.of(TENSOR_25000_INT64),
            4, Set.of(TENSOR_25000_OBJECT),
            5, Set.of(TENSOR_25000_INT64)));
  }

  /**
   * Generator test for {@code tf.strings.as_string} on a 2-arg sink {@code f(y, x)}. Output shape
   * is the input's shape (here {@code (3,)}); output dtype is fixed to {@code string}
   * (wala/ML#422). Asserts that both the {@code as_string} output (`y`, string dtype) at vn=2 and
   * its input (`x`, float32) at vn=3 classify precisely &mdash; the multi-tensor-sink pattern
   * doesn't break classification on either flow when run inside the full test suite.
   */
  @Test
  public void testAsString2ArgSink()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_as_string_2arg_sink.py",
        "f",
        2,
        2,
        Map.of(2, Set.of(TENSOR_3_STRING), 3, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Sibling of {@link #testAsString2ArgSink()} with two 1-arg sinks {@code f(y)} and {@code g(x)},
   * asserting the {@code y}-side sink. The {@code as_string} output's string-dtype tensor
   * classifies precisely at vn=2.
   */
  @Test
  public void testAsStringTwoSinksY()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_as_string_two_sinks.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_STRING)));
  }

  /**
   * Companion to {@link #testAsStringTwoSinksY()} asserting the {@code x}-side sink {@code g(x)}.
   * The {@code x} input classifies precisely as float32 at vn=2.
   */
  @Test
  public void testAsStringTwoSinksX()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_as_string_two_sinks.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testReduceSum2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceSum3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum.py", "h", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * Regression test for `wala/ML#447`: scalar `tf.constant(<bool>)` exercises the {@link
   * java.lang.Boolean} arm of {@code TensorGenerator.getDTypesOfValue}. Without it, dtype inference
   * threw {@code IllegalStateException: Unknown constant type: class java.lang.Boolean}.
   */
  @Test
  public void testBoolConstant()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_bool_constant.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_BOOL)));
  }

  /**
   * List-of-bool form of {@link #testBoolConstant} — exercises the recursive {@code
   * getDTypesOfValue} call (line 1625) on a list whose elements are `Boolean` constants.
   */
  @Test
  public void testBoolConstant2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_bool_constant.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_3_BOOL)));
  }

  @Test
  public void testReduceSum4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum_kwargs.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceSum5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum_kwargs.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceSum6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum_kwargs.py", "h", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Exercises the {@code tf.math.reduce_sum} (ref="math") binding via {@code
   * tf.math.reduce_sum(input_tensor=x, axis=1, keepdims=True)}.
   */
  @Test
  public void testReduceSum7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum_kwargs.py", "i", 1, 1, Map.of(2, Set.of(TENSOR_2_1_FLOAT32)));
  }

  @Test
  public void testReduceSum8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum_kwargs.py", "j", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testGradient()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gradient.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_FLOAT32)));
  }

  @Test
  public void testGradient2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gradient2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_FLOAT32)));
  }

  /**
   * Regression test for <a href="https://github.com/wala/ML/issues/464">wala/ML#464</a>: when
   * {@code sources} is a list (the common Keras pattern), {@code tape.gradient} returns a parallel
   * list of fresh tensors and {@code grads[i]} must resolve to the shape/dtype of the i-th source.
   * The fixture passes both {@code grads[0]} (for {@code w1}, a {@code [2]}-shaped float32) and
   * {@code grads[1]} (for {@code w2}, a {@code [1, 1]}-shaped float32) to {@code f} across two
   * separate calls; with the {@link com.ibm.wala.cast.python.ml.client.Gradient} {@code
   * TupleElementProvider} implementation, {@code f}'s parameter resolves to the union of {@link
   * #TENSOR_2_FLOAT32} and {@link #TENSOR_1_1_FLOAT32}.
   */
  @Test
  public void testGradientList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gradient_list.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_FLOAT32, TENSOR_1_1_FLOAT32)));
  }

  /**
   * Tighter variant of {@link #testGradientList()}: passes both gradients in a single {@code
   * f(grads[0], grads[1])} call, so the analyzer must resolve each argument's tensor type
   * independently per its source index rather than as a union across two call sites. {@code f}'s
   * first parameter (vn=2) must resolve to {@link #TENSOR_2_FLOAT32} (from {@code w1}) and the
   * second parameter (vn=3) to {@link #TENSOR_1_1_FLOAT32} (from {@code w2}). Closes part of <a
   * href="https://github.com/wala/ML/issues/464">wala/ML#464</a>.
   */
  @Test
  public void testGradientList2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gradient_list2.py",
        "f",
        2,
        2,
        Map.of(2, Set.of(TENSOR_2_FLOAT32), 3, Set.of(TENSOR_1_1_FLOAT32)));
  }

  /**
   * Regression test for the wala/ML#518 throw path in {@link
   * com.ibm.wala.cast.python.ml.client.RaggedFromNestedValueRowIds#getShapes}'s {@code
   * nested_nrows} arg-collection loop: when the arg contains a non-numeric string {@link
   * com.ibm.wala.ipa.callgraph.propagation.ConstantKey}, the {@code Long.parseLong((String) val)}
   * site catches {@link NumberFormatException} and rethrows as {@link IllegalStateException} (with
   * the original NFE as {@code cause}). The test exercises that branch by passing {@code
   * nested_nrows=["abc"]} in the Python fixture, and {@code assertThrows} captures the rethrow so
   * the test can assert on the cause—tighter than a bare {@code @Test(expected = …)}, which would
   * pass on any {@code IllegalStateException} raised during analysis regardless of origin. Closes
   * part of <a href="https://github.com/wala/ML/issues/520">wala/ML#520</a> (the {@code
   * RaggedFromNestedValueRowIds} portion).
   */
  @Test
  public void testRaggedNrowsNonNumeric() {
    IllegalStateException ise =
        assertThrows(
            IllegalStateException.class,
            () -> test("tf2_test_ragged_nrows_non_numeric.py", "f", 1, 1, Map.of()));
    assertTrue(
        "Expected cause to be NumberFormatException; got " + ise.getCause(),
        ise.getCause() instanceof NumberFormatException);
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

  /**
   * {@code tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)} returns a fresh loss
   * tensor of shape {@code logits.shape[:-1]} and dtype {@code float32}. For this test, logits is
   * {@code (3, 4) float32} so the loss is {@code (3,) float32} (verified by out-of-band TF runtime
   * probe).
   *
   * <p>Expectation evolution:
   *
   * <ul>
   *   <li>Master: {@code MNIST_INPUT} &mdash; a generic tensor sentinel from before the analyzer
   *       was specific enough to narrow to a rank-1 shape for this sink.
   *   <li>Earlier on branch 267 ({@code 13c7ec0a}): narrowed to {@code TENSOR_3_INT32}, matching
   *       the pass-through bug's behaviour &mdash; the XML {@code <return value="labels"/>} on
   *       {@code sparse_softmax_cross_entropy_with_logits.do()} made the call's result share {@code
   *       labels}' shape and dtype ({@code (3,) int32}). Specific but semantically wrong.
   *   <li>Now (post-wala/ML#412, {@code 0cfdadc4}): {@code TENSOR_3_FLOAT32}. Runtime-correct.
   * </ul>
   */
  @Test
  public void testSparseSoftmaxCrossEntropyWithLogits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_softmax_cross_entropy_with_logits.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testRelu()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_relu.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  // Tier-A pure-passthrough math ops (wala/ML#422). Each is shape and dtype passthrough on `x`;
  // routes to a per-op subclass of `PassThroughUnaryTensorGenerator`. The point of dedicated
  // generators (vs. leaving these on `ReadDataFallback`) is dtype propagation: without these,
  // `tf.math.sqrt(x)` etc. produce ⊤/UNKNOWN, blocking downstream dtype-axis precision through
  // any function whose parameters flow from these ops.

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Sqrt}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSqrt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sqrt.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Log}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLog()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_log.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Negative}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testNegative()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_negative.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Sin}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSin()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sin.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Cos}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCos()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_cos.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Floor}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testFloor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_floor.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Ceil}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCeil()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ceil.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testCeil} that exercises the {@code x} side of the fixture (the input to
   * {@code tf.math.ceil}, asserted at the second sink {@code g(x)}). Combined with {@link
   * #testCeil} (the {@code y} side, the {@code ceil} output), this covers both ends of the
   * passthrough.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCeilInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ceil.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testCeil}: {@code tf.math.ceil(x=x)}. Exercises the keyword
   * arg-resolution path on the {@code PassThroughUnaryTensorGenerator} base.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCeilKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ceil_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testCeilKw} that exercises the {@code x} side of the keyword-arg fixture.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCeilKwInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ceil_kw.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * 2-arg-sink variant of {@link #testCeil}: same input and op, but with one combined sink {@code
   * f(y, x)} instead of two separate single-arg sinks {@code f(y); g(x)}. This is the same shape as
   * the wala/ML#495 multi-tensor-sink pattern; the difference is that #495 is specifically about
   * dataset-loader outputs (`fashion_mnist`/`cifar100`/etc.) flowing through {@link
   * com.ibm.wala.cast.python.ml.client.TensorGenerator#shapesFromSSAChain}'s fallback path. For
   * {@code ceil} on {@code tf.constant}, no fallback is involved, so the pattern works precisely
   * today &mdash; this test asserts the lattice-correct {@code (3,) float32} on both params and
   * stands as a canary: if #495 ever generalizes beyond dataset loaders to per-op generators, this
   * test will start failing.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCeilPair()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ceil_pair.py",
        "f",
        2,
        2,
        Map.of(2, Set.of(TENSOR_3_FLOAT32), 3, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Sign}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSign()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_sign.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.cast(x, dtype)}. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Cast} generator extends {@link
   * com.ibm.wala.cast.python.ml.client.PassThroughUnaryTensorGenerator} for shape and overrides the
   * dtype-arg position to point at {@code dtype}; the {@code tf.cast} {@code pass_through} alias
   * that previously bypassed the override was removed in <a
   * href="https://github.com/wala/ML/issues/499">wala/ML#499</a>, so the static analysis now
   * reports the explicit cast target ({@code int32}) rather than the input's dtype ({@code
   * float32}).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCast()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_cast.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/509">wala/ML#509</a>: a
   * user-defined class that happens to define a {@code set_shape} method must not be classified as
   * a tensor by the static analysis. The {@code set_shape} recognition path must restrict pinning
   * to actual tensor types and let non-tensor receivers fall through untouched.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSetShapeNonTensorReceiver()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_set_shape_non_tensor.py", "consume", 0, 0, Map.of());
  }

  /**
   * Generator-dispatch test for {@code tf.expand_dims(input, axis)}. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.ExpandDims} generator overrides {@code getDefaultShapes} to
   * ⊤ pending an axis-aware shape composer. Replacing the stale {@code array_ops.expand_dims}
   * pass_through alias with the dedicated routing (this PR's earlier review fix) made the override
   * actually fire, so the assertion now sees the ⊤ output the override emits.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/481">wala/ML#481</a>): once the axis-aware
   * composer lands as a follow-up, tighten this from {@code TENSOR_UNKNOWN_SHAPE_FLOAT32} to {@code
   * (1, 3)} float32 (the precise insert-at-axis result).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testExpandDims()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_expand_dims.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testExpandDims()} for {@code axis=-1}: trailing length-1 dim. Input shape
   * {@code (3,)} produces output shape {@code (3, 1)}.
   */
  @Test
  public void testExpandDimsAxisNeg1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_expand_dims_axis_neg1.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_1_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.clip_by_value(t, clip_value_min, clip_value_max)}. Pure
   * passthrough — output shape and dtype both inherit from {@code t}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testClipByValue()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_clip_by_value.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.nn.leaky_relu(features)}. Pure passthrough — output shape
   * and dtype both inherit from {@code features} (the input tensor). Mirrors {@link #testRelu()};
   * the {@link com.ibm.wala.cast.python.ml.client.LeakyRelu} generator extends {@link
   * com.ibm.wala.cast.python.ml.client.PassThroughUnaryTensorGenerator}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLeakyRelu()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_leaky_relu.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.math.pow(x, y)}. Element-wise binary; output shape is the
   * broadcast of {@code x} and {@code y} (here both {@code (3,)}, so {@code (3,)}); output dtype
   * matches {@code x} (TF requires {@code x}/{@code y} to share dtype, so dtype-from-{@code x} is
   * sound). Routed through {@link com.ibm.wala.cast.python.ml.client.ElementWiseOperation}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPow()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_pow.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testPow}: {@code tf.math.pow(x=x, y=y)}. Exercises the
   * keyword arg-resolution path through {@link
   * com.ibm.wala.cast.python.ml.client.ElementWiseOperation}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPowKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_pow_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Mixed positional/keyword variant of {@link #testPow}: {@code tf.math.pow(x, y=y)}. Exercises
   * the case where the first argument is positional and the rest are keyword arguments.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPowMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_pow_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  @Test
  public void testRange()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Regression test for wala/ML#451 (reproducer 3): each element {@code i} from {@code for i in
   * tf.range(...)} is a 0-D scalar int32 tensor. The receiving function {@code f}'s parameter must
   * be tensor-classified — never primitive-co-classified downstream. Differs from {@link
   * #testRange()} in being a stripped-down fixture that mirrors the issue body verbatim (no
   * intermediate {@code start}/{@code limit}/{@code delta} variables, no Python {@code assert}s
   * intervening between the {@code tf.range} call and the {@code for}-loop). Pre-fix, certain binop
   * sources at iteration time could shadow the element's tensor classification with spurious
   * primitive entries (the same {@link ElementWiseOperation} over-dispatch fixed by the binop
   * operand-tensor gate); the cleaner fixture here exercises that path directly.
   */
  @Test
  public void testRangeIterationElementType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_iter.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testRange2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range2.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testRange3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range3.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testRange4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_range4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testRange5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_INT32)));
  }

  /**
   * Regression test for wala/ML#492. When an explicit {@code dtype=} keyword is supplied to {@code
   * tf.range}, the analyzer honors it instead of defaulting to {@code int32}. {@code tf.range(0, 5,
   * dtype=tf.float32)} infers {@code float32} via {@link Range#getDTypes}'s dtype-arg dispatch.
   */
  @Test
  public void testRangeDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_dtype.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  /**
   * Companion to {@link #testRangeDType()} — explicit {@code dtype=tf.int64} should be honored (not
   * collapsed to the {@code int32} default). Pinpoints that the dtype-arg path resolves arbitrary
   * {@link com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType} values, not just {@code
   * float32}.
   */
  @Test
  public void testRangeDTypeInt64()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_dtype_int64.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT64)));
  }

  /**
   * Companion to {@link #testRangeDType()} — 1-positional form {@code tf.range(limit, dtype=...)}.
   * Verifies that the dtype-arg dispatch fires regardless of how many positional args are present
   * (the {@link Range} class's call-string-based shape resolution is independent of dtype lookup).
   */
  @Test
  public void testRangeDType1Arg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_dtype_1arg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  /**
   * Regression test for the implicit-dtype path of {@link Range}: when no explicit {@code dtype} is
   * supplied but the {@code start}/{@code limit}/{@code delta} arguments are {@code float}-typed,
   * TF promotes the output to {@code float32} at runtime. {@link Range#getDefaultDTypes} now
   * derives its result from the numeric argument types, matching that promotion. Fix for <a
   * href="https://github.com/wala/ML/issues/492">wala/ML#492</a>.
   */
  @Test
  public void testRangeFloatArgs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_float_args.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

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
   * Regression guard for the {@code @dataclass} half of <a
   * href="https://github.com/wala/ML/issues/205">wala/ML#205</a>: a module containing a {@code
   * @dataclass} definition loads and analyzes without a front-end parse error. The dataclass is
   * defined but unused in the dataflow; {@code f} receives a tensor directly, so its parameter type
   * is recovered iff the module parsed. Companion to {@link #testModule68}/{@link #testModule69},
   * which guard the same for {@code NamedTuple}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDataclassParse()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataclass_parse.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Verifies tensor types propagate through a user-defined {@code NamedTuple} field (<a
   * href="https://github.com/wala/ML/issues/579">wala/ML#579</a>): a tensor stored in a {@code
   * NamedTuple} field and read back ({@code b = w.tensor}) keeps its {@code (4, 8) float32} type.
   * Unlike {@link #testModule68}/{@link #testModule69} — which only confirm a {@code NamedTuple}
   * <em>definition</em> parses — this exercises actual field dataflow: PEP-526 annotated fields now
   * reach the CAst as ordered field entities (jython3 grammar/AST support), and the synthesized
   * constructor populates them positionally. It is the minimal form of the GCN blocker in
   * wala/ML#570, where {@code GraphConvolution.call} unwraps a {@code GNNInput} {@code NamedTuple}
   * the same way.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNamedTupleFieldRead()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_namedtuple_field.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins the forward result of a nested layer call ({@code self.inner(...)} inside another layer's
   * {@code call}): the inner layer's return is tensor-typed at the nested call site and flows into
   * a sink (<a href="https://github.com/wala/ML/issues/570">wala/ML#570</a>; enabled by the
   * wala/ML#595 forward-result machinery).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNestedLayerCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nested_layer_call.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probe for the indexed sub-layer dispatch shape of <a
   * href="https://github.com/wala/ML/issues/661">wala/ML#661</a>: a plain list of sublayers
   * populated by {@code append} in {@code build}, dispatched through a dynamic subscript ({@code
   * self.sub_layers[i](out, training)}) in {@code call} — the miniature of NLPGNN's {@code
   * GAAELayer.encoder}. The inner layer's {@code call} must exist in the call graph with its {@code
   * inputs} parameter tensor-typed.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIndexedLayerListCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_layer_list.py", "Inner.call", 1, 1, Map.of(3, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Constant-index variant of {@link #testIndexedLayerListCall()} (wala/ML#661): {@code
   * self.sub_layers[0](out, training)} over an append-built list. Pins that the fix does not depend
   * on the subscript being dynamic.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIndexedLayerListCallConst()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_layer_list_const.py", "Inner.call", 1, 1, Map.of(3, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * List-literal variant of {@link #testIndexedLayerListCall()} (wala/ML#661): the sublayer list is
   * built as a literal instead of by {@code append}, so the subscript read resolves through the
   * ordinary numeric field. This passed before the fix and pins the discriminator that localized
   * the gap to the append-contents property.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIndexedLayerListCallLiteral()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_layer_list_lit.py", "Inner.call", 1, 1, Map.of(3, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins the forward result of a nested layer call whose argument is a {@code NamedTuple} and whose
   * inner {@code call} computes through a field read, a matmul, and an {@code unsorted_segment_sum}
   * (<a href="https://github.com/wala/ML/issues/570">wala/ML#570</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNestedLayerCallNamedTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nested_layer_call2.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins the forward result of a nested layer call whose inner {@code call} returns through a
   * second method hop ({@code self.propagate(...)} on the same class), the single-class form of
   * {@code gcn_proj}'s return chain (<a href="https://github.com/wala/ML/issues/570">
   * wala/ML#570</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNestedLayerCallPropagate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nested_layer_call3.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins the forward result of a nested layer call whose inner {@code call} returns through an
   * <em>inherited</em> method ({@code self.propagate(...)} defined on a same-module base class),
   * mirroring {@code gcn_proj}'s {@code GraphConvolution(MessagePassing)} shape (<a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNestedLayerCallInheritedPropagate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nested_layer_call4.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins the forward result of a nested layer call whose inner {@code call} returns through a
   * method inherited from a <em>cross-module</em> base ({@code Inner(MessagePassing)} with {@code
   * MessagePassing} in another file), the deepest structural form of the {@code gcn_proj} chain (<a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a>). {@code consume(x)} inside {@code
   * Outer.call} recovers the concrete {@code (4, 8) float32}, so the frame/inheritance mechanism is
   * not the {@code gcn_proj} residual; the remaining gap is the list-mediated aggregation inside
   * the vendored {@code propagate} (gathers/slices/appends over the adjacency-list collection).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNestedLayerCallCrossModule()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "nested_proj/messagepassing.py",
          "nested_proj/inner.py",
          "nested_proj/tf2_test_nested_cross_module.py"
        },
        "tf2_test_nested_cross_module.py",
        "consume",
        "nested_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Verifies dtype recovery for {@code tf.linalg.matmul} on a {@code NamedTuple} field threaded in
   * as a parameter (<a href="https://github.com/wala/ML/issues/570">wala/ML#570</a>). {@code Inp}
   * is constructed in the caller and passed into {@code layer}, which reads {@code inp.x} (a {@code
   * NamedTuple} field) and feeds it to {@code matmul}; {@code consume(h)} pins the result. The
   * field read has no points-to set at the read site (the {@code NamedTuple} was built in the
   * caller), so the matmul input's dtype is recovered by reading the field off the object's
   * instance in the heap &mdash; the minimal form of the {@code gcn_proj} {@code
   * GraphConvolution.call} inner chain.
   *
   * <p>Before the fix, the field read's empty points-to set left the matmul input ⊤ on both axes;
   * now the dtype is recovered from the instance field and, for this single-call shape, the
   * existing shape machinery resolves the rest, so {@code consume}'s parameter is the concrete
   * {@code (4, 4) float32}. (The harder multi-call/chained case &mdash; {@code gcn_proj}'s layer
   * outputs &mdash; still needs forward-result propagation; see wala/ML#570.)
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeTupleReturnUnpack()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tuple_return_unpack.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  @Test
  public void testCollectionProbeLayerTupleReturn()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_layer_tuple_return.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  @Test
  public void testCollectionProbeListAppendIterate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_list_append_iterate.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Probe for the model-forward tuple-of-reshapes shape: a {@code tf.keras.Model} subclass whose
   * {@code call} returns a tuple of {@code tf.reshape} results, unpacked at the top-level call site
   * and passed to {@code consume}. Discriminates the reshape-producer axis against the passing
   * layer-tuple-return shape ({@link #testCollectionProbeLayerTupleReturn()}).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testModelForwardTupleReshapeReturn()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_tuple_reshape_return.py",
        "consume",
        2,
        2,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32), 3, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Control for {@link #testModelForwardTupleReshapeReturn()}: identical shape but the returned
   * tuple's elements are elementwise results rather than {@code tf.reshape} results.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testModelForwardTupleAddReturn()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_tuple_add_return.py",
        "consume",
        2,
        2,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32), 3, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probe for the generator-fed model-forward shape: as {@link
   * #testModelForwardTupleReshapeReturn()} but the model input arrives via {@code next()} on a
   * generator function, tuple-unpacked at the call site. The generator transit drops the tensor
   * typing (<a href="https://github.com/wala/ML/issues/696">wala/ML#696</a>), so {@code consume}
   * previously saw zero tensor parameters; a regression guard for <a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testModelForwardTupleReshapeGenInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_tuple_reshape_gen_input.py",
        "consume",
        2,
        2,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32), 3, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probe for the bare generator/next transit: a tensor yielded by a generator function, obtained
   * via {@code next()} with tuple unpacking, flows directly to {@code consume} with no model in
   * between; a regression guard for <a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>, where the generator transit
   * dropped the typing.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGenNextUnpack()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gen_next_unpack.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Narrowing probe for the generator transit: the generator yields a single tensor (no tuple),
   * retrieved via {@code next()} with no unpacking. The minimal failing shape of <a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>: neither tuple unpacking nor a
   * model forward is involved. The minimal shape of <a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>, where the generator transit
   * dropped the typing; now a regression guard for it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGenNextSingle()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gen_next_single.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probe for the transit through an explicit {@code iter()} call: {@code next(iter(gen()))}.
   * {@code iter()} was modeled as a fresh, empty {@code iterator} allocation, so {@code next}'s
   * read of the generator content field landed on the empty iterator instead of the generator
   * object and the yielded tensor's type was dropped. A regression guard for <a
   * href="https://github.com/wala/ML/issues/698">wala/ML#698</a>, where {@code iter} is modeled as
   * a pass-through of its argument.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGenNextIter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gen_next_iter.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probe for the two-argument {@code next(it, default)} form: the iterator is an empty generator,
   * so at runtime {@code next} returns the default and its type must reach the result. The
   * default's flow was dropped because the summary read only the iterator's generator content
   * field. A regression guard for <a href="https://github.com/wala/ML/issues/699">wala/ML#699</a>,
   * where the default (arg 3) is unioned into the result through a reachable join.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGenNextDefault()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gen_next_default.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Companion probe for the generator transit: the same yielded pair consumed by for-loop
   * destructuring over the generator instead of {@code next()}. Also dropped (<a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>); distinct from the {@code
   * tf.data} destructuring shape of <a
   * href="https://github.com/wala/ML/issues/396">wala/ML#396</a>, where the producer is modeled and
   * the symptom is swapped element types. A regression guard for <a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>, where the generator transit
   * dropped the typing.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGenForUnpack()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gen_for_unpack.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/668">wala/ML#668</a>: appending
   * a constant (an invariant-contents value whose pointer key the invoke's own argument processing
   * records as implicitly represented) must not crash call-graph construction with {@code
   * UnimplementedError}. The tensor appended alongside still types through the iteration.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeListAppendConstant()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_list_append_constant.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probes the wala/ML#570 residual: a list accumulated with {@code append} in a loop feeds {@code
   * tf.concat} (mirroring {@code MessagePassing._calculate_messages_all_type} feeding {@code
   * _aggregate_function}). The appended values' shapes and dtype survive the list: the result keeps
   * the rank and non-axis dims with a dynamic axis dim (the element count is not statically known)
   * and the float32 dtype.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeListAppendConcat()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_list_append_concat.py",
        "consume",
        1,
        1,
        Map.of(
            2, Set.of(new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(8))))));
  }

  @Test
  public void testCollectionProbeZipIterate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_zip_iterate.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  @Test
  public void testCollectionProbeListElemSlice()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_list_elem_slice.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 4))));
  }

  @Test
  public void testCollectionProbeListLiteralIterate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_list_literal_iterate.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  @Test
  public void testCollectionProbeZipDiag()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_zip_diag.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins the gpt-2 decoder-stack shape in miniature (wala/ML#618): layers built by a list
   * comprehension, iterated with {@code zip} against a {@code [None] * n} list, each call's tuple
   * result destructured and the hidden state carried through the loop.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeLayerListComprehension()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_layer_list_comprehension.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Indexed dispatch over a comprehension-built sublayer list (wala/ML#661 shape 3, wala/ML#694):
   * {@code self.sub_layers = [Inner() for _ in range(n)]} dispatched through {@code
   * self.sub_layers[i](out)} in a loop. {@code Inner.call} returns a distinctly-shaped {@code (6,
   * 1)} tensor, so a working dispatch would flow {@code (6, 1)} to {@code consume}. The analysis
   * instead reports the pre-loop input's {@code (2, 3)} (carried by the loop phi): the
   * comprehension-built indexed call materializes no callee, so the sub-layer forward result never
   * reaches the sink.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/694">wala/ML#694</a>): once the
   * comprehension-built indexed dispatch materializes its callee, tighten the assertion to the
   * precise {@code (6, 1)} shape (the Python runtime shape).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testIndexedComprehensionLayerListCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // TODO(wala/ML#694): observed-but-imprecise (2, 3); the runtime shape is (6, 1).
    test("tf2_test_layer_list_compr.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins {@code self.add_weight(...)} (wala/ML#618): the Keras weight-creation API, called from the
   * lazily-invoked {@code build} (wala/ML#595), creates a tensor whose shape and dtype come from
   * the call's {@code shape} list and {@code dtype} string arguments (wala/ML#667), so the matmul
   * against it composes to {@code (4, 4)} float32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeAddWeight()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add_weight.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the {@code add_weight} result itself (wala/ML#667): the weight's shape comes from the
   * {@code shape} list argument and its dtype from a {@code tf.float32} module-constant argument
   * (the string form is covered by {@link #testCollectionProbeAddWeight()}).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testAddWeightArguments()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add_weight2.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/672">wala/ML#672</a>: {@code add_weight}
   * without a {@code dtype} argument follows Keras's documented default and types float32 (the
   * layer variable dtype under the default global policy), via the allocator's float32 default.
   * Completes the dtype-form trio with {@link #testCollectionProbeAddWeight()} (string) and {@link
   * #testAddWeightArguments()} (module constant).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testAddWeightDefaultDtype()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_add_weight3.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * Pins the vendored gpt-2 {@code EmbeddingLayer} forward result (wala/ML#618): the {@code
   * add_weight}-created weight dispatches and both {@code mode} branches contribute to the result
   * union. With {@code add_weight} consuming its {@code shape}/{@code dtype} arguments
   * (wala/ML#667), the embedding-mode member is fully concrete: {@code (2, 3, 8)} float32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeVendoredEmbedding()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // Count-only: the parameter's type is a union across the `mode` branches whose exact
    // members shift with modeling precision.
    test(
        new String[] {
          "gpt2_vendored/layers/__init__.py",
          "gpt2_vendored/layers/embedding_layer.py",
          "gpt2_vendored/probe_embedding.py"
        },
        "probe_embedding.py",
        "consume",
        "gpt2_vendored",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 3, 8),
                new TensorType(
                    INT_32, asList(DynamicDim.INSTANCE, DynamicDim.INSTANCE, new NumericDim(10))),
                new TensorType(
                    INT_32,
                    asList(new SymbolicDim("?"), new SymbolicDim("?"), new SymbolicDim("?"))))));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/683">wala/ML#683</a>: {@code
   * self._distribution_strategy} is Keras-internal state assigned by {@code Model.__init__}, never
   * in user code. With the shell {@code Model.__init__} modeling the attribute, the receiver of
   * {@code self._distribution_strategy.run(self.__train_step, args=(x, y))} resolves to the
   * strategy instance and the {@code run} summary materializes the invoke edge, typing both
   * callback parameters through the args-tuple forwarding (wala/ML#618).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDistTrainStep()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dist_train_step.py",
        "MyModel.__train_step",
        2,
        3,
        Map.of(3, Set.of(TensorType.of(FLOAT_32, 2, 3)), 4, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/683">wala/ML#683</a> at subject shape
   * (MusicTransformer-tensorflow2.0): the model base is {@code keras.Model} bound by {@code from
   * tensorflow.python import keras}, so the summary {@code Model} must be reachable through the
   * {@code tensorflow.python} module object for the class shell to carry {@code Model.__init__}'s
   * {@code _distribution_strategy} assignment; and the callback args tuple has four elements, so
   * the strategy {@code run} summary must forward past the first two.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDistTrainStep2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dist_train_step2.py",
        "MyModel.__train_step",
        3,
        4,
        Map.of(
            3,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            4,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            5,
            Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/683">wala/ML#683</a> at the
   * MusicTransformer-tensorflow2.0 encoder-decoder shape: the callback args tuple has seven
   * elements, the widest the subject passes to the strategy, so the {@code run} summary's tuple
   * forwarding must reach fields 2 through 6. The fixture's callback computes from the tuple's
   * fifth and sixth elements specifically, so the pinned result only types if the later fields
   * flow.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDistTrainStep3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dist_train_step3.py",
        "MyModel.__train_step",
        6,
        7,
        Map.of(
            3,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            4,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            5,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            6,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            7,
            Set.of(TensorType.of(FLOAT_32, 2, 3)),
            8,
            Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/684">wala/ML#684</a> at subject scale: in the
   * vendored MusicTransformer-tensorflow2.0 whole-project analysis, {@code
   * MusicTransformer.__prepare_train_data}'s direct {@code tf.ones((y.shape[0], 1), dtype=y.dtype)}
   * must be visible through the {@code tf} binding carried by {@code from custom.layers import *}
   * (a wildcard re-export of an import binding). What is asserted is the function's tensor-variable
   * census: seven distinct value numbers, which include the {@code tf.ones} result and its
   * downstream locals (observed as the runtime-true {@code (Dynamic, 1)} of float32 at vn 6);
   * pre-fix, only the two ⊤-typed parameters survived, so any regression of the binding collapses
   * the count. The vendored file is verbatim, so no {@code consume} sink can pin the local's exact
   * type here; the exact-type pins live in the fixture-scale probes ({@link #testWildcardUsedTf()},
   * {@link #testWildcardPkgNoInitTf()}). (The sibling {@code
   * MusicTransformerDecoder.__prepare_train_data} has no live {@code tf} use; all its tensor lines
   * are commented out in the subject.) Before the wala/ML#683 {@code tensorflow.python} namespace
   * binding, the empty {@code keras} binding that {@code custom/layers.py} re-exports starved the
   * whole chain.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testMusicTransformerPrepareTrainData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        MUSICTRANSFORMER_PROJECT_FILES,
        "model.py",
        "MusicTransformer.__prepare_train_data",
        "musictransformer_proj",
        2,
        7,
        Map.of(
            2,
            Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE),
            3,
            Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Pins the model self-call (wala/ML#618): a method calling {@code self(...)} and destructuring
   * the tuple result, mirroring gpt-2's {@code predictions, _ = self(inputs, training=True)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeModelSelfCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_model_self_call.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the vendored gpt-2 forward output (the wala/ML#618 {@code pred} source): with wala/ML#665
   * forwarding wildcard import bindings, the full decoder stack types and the model output is a
   * rank-3-dominated tensor union. Receiver-keyed trampoline contexts (wala/ML#679) removed the
   * spurious {@code (?, ?, 4)}/{@code (?, ?, 12)} constructor-collapse members and kept the
   * runtime-true vocab member {@code (?, ?, 10)}, but the previously concrete {@code (2, 3, 8)
   * float32} member degrades to {@code ? of float32} in decoder-stack propagation ({@link
   * #testCollectionProbeVendoredEmbedding()} still recovers it concretely at the embedding output).
   *
   * <p>TODO: Expect the float32 member's concrete shape back once <a
   * href="https://github.com/wala/ML/issues/682">wala/ML#682</a> recovers concrete shapes through
   * the decoder stack under receiver-keyed contexts.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeVendoredForward()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_vendored/layers/__init__.py", "gpt2_vendored/layers/embedding_layer.py",
          "gpt2_vendored/layers/feed_forward.py", "gpt2_vendored/layers/layer_norm.py",
          "gpt2_vendored/layers/attention_layer.py", "gpt2_vendored/utils/__init__.py",
          "gpt2_vendored/utils/tf_utils.py", "gpt2_vendored/scripts/__init__.py",
          "gpt2_vendored/scripts/utils.py", "gpt2_vendored/data_pipeline.py",
          "gpt2_vendored/A.py", "gpt2_vendored/probe_forward.py"
        },
        "probe_forward.py",
        "consume",
        "gpt2_vendored",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TENSOR_UNKNOWN_SHAPE_FLOAT32,
                new TensorType(
                    UNKNOWN,
                    asList(new SymbolicDim("?"), new SymbolicDim("?"), new SymbolicDim("?"))),
                new TensorType(
                    UNKNOWN,
                    asList(DynamicDim.INSTANCE, DynamicDim.INSTANCE, new NumericDim(10))))));
  }

  /**
   * Pins wala/ML#670's fixes directly: {@code GlobalAveragePooling1D} is modeled (rank-3 input,
   * temporal axis dropped), so the functional model's weight walk resolves the downstream {@code
   * Dense} kernel {@code (8, 5)} and bias {@code (5,)} concretely.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGap1dWeights()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gap1d_weights.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 8, 5), TENSOR_5_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/669">wala/ML#669</a>: {@code
   * build_model} is vendored verbatim from {@code LongmaoTeamTf/deep_recommenders} ({@code
   * examples/train_transformer_on_imdb_keras.py}), a functional {@code tf.keras.Model} whose
   * weight-graph walk ({@code Model.getWeightShapes}) resolves {@code Dense}/{@code MatMul} weight
   * shapes — the exact frames that crashed with {@code IllegalStateException} on 0.52.12 when a
   * WALA 1.8.0 non-constant key reached {@code getConstantValues}. The crash's non-constant {@code
   * units} key itself arises only under the consumer-side speculative call-graph configuration,
   * which this harness does not enable, so this guard pins that the walk completes; the {@code
   * getConstantValues} degrade contract is exercised structurally.
   *
   * <p>With wala/ML#670 fixed, the walk traverses past the head {@code Dense} into the transformer
   * (an unresolvable input no longer stops the trace-back, and {@code GlobalAveragePooling1D} is
   * modeled — see {@link #testGap1dWeights()}), but it still yields no weight shapes here: the
   * pooling input is the vendored transformer's forward output, whose shape is the wala/ML#570
   * residual. TODO: expect the concrete weight-shape union once <a
   * href="https://github.com/wala/ML/issues/570">wala/ML#570</a> is fixed.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTransformerWeights()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "tr_proj/deep_recommenders/__init__.py",
          "tr_proj/deep_recommenders/keras/__init__.py",
          "tr_proj/deep_recommenders/keras/models/__init__.py",
          "tr_proj/deep_recommenders/keras/models/nlp/__init__.py",
          "tr_proj/deep_recommenders/keras/models/nlp/multi_head_attention.py",
          "tr_proj/deep_recommenders/keras/models/nlp/transformer.py",
          "tr_proj/tf2_test_transformer_weights.py"
        },
        "tf2_test_transformer_weights.py",
        "consume",
        "tr_proj",
        0,
        0,
        emptyMap());
  }

  /**
   * Pins a tensor computed inside a {@code with tf.name_scope(...)} block (wala/ML#618): the
   * unresolved context manager does not perturb the body's dataflow.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeWithNameScope()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_with_name_scope.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the vendored gpt-2 {@code Conv1d} forward result: {@code tf} arrives through the wildcard
   * import (wala/ML#665) and the {@code add_weight}-built kernel dispatches, so the result is
   * tensor-classified.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeVendoredConv1d()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_vendored/layers/__init__.py", "gpt2_vendored/layers/embedding_layer.py",
          "gpt2_vendored/layers/feed_forward.py", "gpt2_vendored/layers/layer_norm.py",
          "gpt2_vendored/layers/attention_layer.py", "gpt2_vendored/utils/__init__.py",
          "gpt2_vendored/utils/tf_utils.py", "gpt2_vendored/scripts/__init__.py",
          "gpt2_vendored/scripts/utils.py", "gpt2_vendored/data_pipeline.py",
          "gpt2_vendored/A.py", "gpt2_vendored/probe_conv1d.py"
        },
        "probe_conv1d.py",
        "consume",
        "gpt2_vendored",
        1,
        1,
        // The computed output shape is opaque (a runtime-built list), so the reshape result is
        // pinned at unknown rank (wala/ML#703); receiver-keyed contexts (wala/ML#679) recover the
        // `add_weight` float32 dtype.
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Pins {@code tf.reshape} with a literal shape list (companion to {@link
   * #testCollectionProbeReshapeComputedShape()}).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeReshapeLiteralShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_computed_shape.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT32, 6, 8))));
  }

  /**
   * Pins {@code tf.reshape} with a shape list built at runtime by list concatenation ({@code
   * [tf.shape(x)[0], tf.shape(x)[1]] + [n]}, the vendored {@code Conv1d} idiom, wala/ML#618): the
   * interpreter evaluates the expression and the output shape is concrete.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeReshapeComputedShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_computed_shape.py",
        "consume2",
        1,
        1,
        // The interpreter evaluates the concatenated shape expression to the concrete
        // (2, 3, 16); the opaque-shape-operand unknown-rank pin (wala/ML#703) rides along in the
        // union. TODO(https://github.com/wala/ML/issues/703): drop the ⊤ member once the pin
        // defers to interpreter-resolved shape operands.
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3, 16), TENSOR_UNKNOWN_SHAPE_FLOAT32)));
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
   * Pins wala/ML#665: {@code tf} reached through {@code from helpers import *} binds, matching
   * Python's wildcard semantics (every public module-level name is exported, including modules the
   * source module imported).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeWildcardTf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"wildcard_proj/helpers.py", "wildcard_proj/tf2_test_wildcard_tf.py"},
        "tf2_test_wildcard_tf.py",
        "consume",
        "wildcard_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probe for <a href="https://github.com/wala/ML/issues/684">wala/ML#684</a>: {@code tf} arrives
   * through {@code from helpers import *} in a script with no direct tensorflow import, and is read
   * inside a name-mangled {@code @staticmethod} of a {@code tf.keras.Model} subclass invoked
   * self-qualified — the subject's {@code MusicTransformer.__prepare_train_data} shape, several
   * levels deeper than {@link #testCollectionProbeWildcardTf()}'s script-level read. The wildcard
   * binding resolves, the shape is concrete, and the {@code dtype=y.dtype} attribute argument is
   * consumed (wala/ML#686), so the result is the runtime-true {@code (2, 1) int32}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testWildcardMethodTf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"wildcard_proj/helpers.py", "wildcard_proj/tf2_test_wildcard_method_tf.py"},
        "tf2_test_wildcard_method_tf.py",
        "consume",
        "wildcard_proj",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 2, 1))));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/684">wala/ML#684</a> at fixture scale: {@code
   * tf} reached through {@code from helpers_used import *} where the exporting module also reads
   * {@code tf} inside one of its own functions. The intra-module use lexically exposes the binding,
   * which drops it from the script body's SSA local names, so the wildcard scan's named-binding
   * match (wala/ML#665) must consult the exposed-name information as well. The subject's {@code
   * custom/layers.py} has this shape; the untouched-binding {@code helpers.py} probes do not.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testWildcardUsedTf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "wildcard_proj/helpers_used.py", "wildcard_proj/tf2_test_wildcard_used_tf.py"
        },
        "tf2_test_wildcard_used_tf.py",
        "consume",
        "wildcard_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins <a href="https://github.com/wala/ML/issues/684">wala/ML#684</a> at fixture scale for the
   * package-qualified form: {@code tf} reached through {@code from pkgnoinit.helpers3 import *},
   * where the package has no {@code __init__.py} (a namespace package, like the subject's {@code
   * custom/}) and the exporting module also reads {@code tf} in one of its own functions — the
   * exact {@code from custom.layers import *} shape of MusicTransformer-tensorflow2.0.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testWildcardPkgNoInitTf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "wildcard_proj/pkgnoinit/helpers3.py", "wildcard_proj/tf2_test_wildcard_pkgnoinit_tf.py"
        },
        "tf2_test_wildcard_pkgnoinit_tf.py",
        "consume",
        "wildcard_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the vendored {@code LayerNormalization} forward result: {@code add_weight}-created {@code
   * gamma}/{@code beta} dispatch (wala/ML#595, wala/ML#618) and the normalization body types.
   * Receiver-keyed contexts (wala/ML#679) dropped the shapeless-and-dtypeless union member, so the
   * result is exactly {@code ? of float32}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testVendoredLayerNorm()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_vendored/layers/__init__.py", "gpt2_vendored/layers/embedding_layer.py",
          "gpt2_vendored/layers/feed_forward.py", "gpt2_vendored/layers/layer_norm.py",
          "gpt2_vendored/layers/attention_layer.py", "gpt2_vendored/utils/__init__.py",
          "gpt2_vendored/utils/tf_utils.py", "gpt2_vendored/scripts/__init__.py",
          "gpt2_vendored/scripts/utils.py", "gpt2_vendored/data_pipeline.py",
          "gpt2_vendored/A.py", "gpt2_vendored/probe_ln.py"
        },
        "probe_ln.py",
        "consume",
        "gpt2_vendored",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Companion to {@link #testCollectionProbeWildcardTf()} with a package-qualified wildcard source
   * ({@code from pkg.helpers2 import *}), the vendored {@code feed_forward.py} form (wala/ML#665).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeWildcardPkgTf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "wildcard_proj/pkg/__init__.py",
          "wildcard_proj/pkg/helpers2.py",
          "wildcard_proj/tf2_test_wildcard_pkg_tf.py"
        },
        "tf2_test_wildcard_pkg_tf.py",
        "consume",
        "wildcard_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probes wala/ML#661's indexed sub-layer call shape ({@code self.container[i](x)}).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIndexedLayerCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_indexed_layer_call.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probes wala/ML#666's dotted-alias case ({@code import tensorflow.keras.backend as K} read
   * inside a method).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testBackendAliasCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_backend_alias.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  @Test
  public void testNamedTupleFieldMatmul()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_namedtuple_field_matmul.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Same as {@link #testNamedTupleFieldRead} but the {@code NamedTuple} base is written as the
   * dotted attribute chain {@code typing.NamedTuple} rather than a bare {@code NamedTuple}. This
   * guards the dotted-base path of {@code PythonConstructorTargetSelector.isPositionalFieldClass}:
   * the front-end must record the full {@code typing.NamedTuple} supertype name (not just the root
   * {@code typing}) for the positional-field synthesis to fire, so the tensor stored in the field
   * and read back ({@code b = w.tensor}) keeps its {@code (4, 8) float32} type. Without the full
   * dotted-name capture the supertype collapses to {@code typing}, {@code isPositionalFieldClass}
   * returns {@code false}, and {@code consume} sees zero tensor parameters (<a
   * href="https://github.com/wala/ML/issues/571">wala/ML#571</a>).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNamedTupleFieldReadDotted()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_namedtuple_field_dotted.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Verifies tensor types propagate through a {@code typing.Tuple}-annotated tuple-of-tensors
   * parameter: a 2-tuple of tensors passed to {@code f} and unpacked ({@code x, y = inputs}) keeps
   * each element's type, so {@code consume(x)} sees {@code (4, 8) float32}. This mirrors the
   * perf-eval corpus's {@code deep_recommenders} {@code CIN.call(self, inputs: Tuple[tf.Tensor,
   * tf.Tensor])} — an {@code @tf.function}-decorated function the Hybridize tool refactors — and is
   * the tuple-parameter analogue of {@link #testNamedTupleFieldRead}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTupleParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tuple_param.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/581">wala/ML#581</a>: a {@code
   * tf.reshape} whose dim is arithmetic over instance attributes ({@code self.heads *
   * self.out_features}) infers the precise {@code (4, 512)}. Both shape-argument extraction paths
   * now fold the binary op over constant-valued field reads through the points-to analysis (the
   * shared {@link com.ibm.wala.cast.python.ml.types.TensorType#foldArithmeticDim}): the
   * generator-side {@code getShapesFromShapeArgument} and the interpreter-based {@code
   * TensorType.shapeArg}, which previously degraded to {@code DynamicDim} / {@code SymbolicDim}
   * respectively since {@code interpretAsInt} cannot evaluate {@code self.X} as source text.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testReshapeSelfArith()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_self_arith.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_512_FLOAT32)));
  }

  /**
   * Coverage companion to {@link #testReshapeSelfArith()} for <a
   * href="https://github.com/wala/ML/issues/581">wala/ML#581</a>: a {@code tf.reshape} dim of
   * {@code self.base + 4} infers the precise {@code (2, 64)}. Exercises the {@code ADD} operator
   * and a literal operand of the arithmetic fold (the sibling fixture covers {@code MUL} over two
   * field reads), so {@link com.ibm.wala.cast.python.ml.types.TensorType#resolveConstantInt}
   * resolves one operand via the symbol table ({@code 4}) and the other via the points-to analysis
   * ({@code self.base}).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testReshapeSelfArithAdd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_self_arith_add.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_64_FLOAT32)));
  }

  /**
   * Verifies a module-level PEP-526 annotated assignment with a value (`t: tf.Tensor = tf.ones([2,
   * 3])`) declares its target and propagates the value (wala/ML#579). Outside a class body
   * `visitAnnAssign` must declare a simple-name target like `visitAssign` does; otherwise the
   * target is left undeclared. The tensor flows to `consume` as `(2, 3) float32`.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testAnnAssignLocal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_annassign_local.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Verifies a class-body PEP-526 annotated assignment with a value ({@code weight: tf.Tensor =
   * tf.ones([3, 4])}) assigns the class attribute and that reading it back ({@code y = C.weight})
   * recovers the {@code (3, 4) float32} type (<a
   * href="https://github.com/wala/ML/issues/579">wala/ML#579</a>). This guards the value-bearing
   * branch of {@code visitAnnAssign}: unlike an annotation-only declaration ({@code x: T}, which
   * declares the field but emits no member {@code put}), a value-bearing class field still emits
   * the {@code put}, so the attribute is typed.
   */
  @Test
  public void testClassAttrAnnAssign()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_classattr_annassign.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_3_4_FLOAT32)));
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

  @Test
  public void testStaticMethod() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod2() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method2.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod3() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method3.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod4() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method4.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod5() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method5.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod6() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method6.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod7() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method7.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod8() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method8.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod9() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method9.py",
        "MyClass.the_static_method",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod10() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method10.py",
        "MyClass.the_static_method",
        2,
        2,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod11() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_static_method11.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod12() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_static_method12.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testStaticMethod13() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method13.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testStaticMethod14() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_static_method14.py",
        "MyClass.the_static_method",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testClassMethod() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_class_method.py",
        "MyClass.the_class_method",
        1,
        1,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testClassMethod2() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_class_method2.py",
        "MyClass.the_class_method",
        1,
        1,
        Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testClassMethod3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_class_method3.py", "MyClass.f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testClassMethod4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_class_method4.py", "MyClass.f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testClassMethod5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_class_method5.py", "MyClass.f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testAbstractMethod() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_abstract_method.py", "D.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_abstract_method.py", "C.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testAbstractMethod2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_abstract_method2.py", "D.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_abstract_method2.py", "C.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testAbstractMethod3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_abstract_method3.py", "C.f", 1, 1, Map.of(3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test https://github.com/wala/ML/issues/188.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#188 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method2.py", "f", 1, 1, Map.of(2, Set.of(MNIST_INPUT)));
  }

  /**
   * Test https://github.com/wala/ML/issues/190.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#190 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method3.py", "raffi", 1, 1, Map.of(2, Set.of(MNIST_INPUT)));
  }

  @Test
  public void testDecoratedMethod4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method4.py", "raffi", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method5.py", "raffi", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod6() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method6.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod7() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method7.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod8() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method8.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * This decorator isn't defined. Thus, we shouldn't have a CG node for it.
   *
   * <p>We now require nodes for functions under test. Otherwise, a test could pass even though the
   * function doesn't exist.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod9() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method9.py", "f", 0, 0);
  }

  /**
   * Test https://github.com/wala/ML/issues/190.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#190 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod10() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method10.py", "f", 1, 1, Map.of(2, Set.of(MNIST_INPUT)));
  }

  @Test
  public void testDecoratedMethod11() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method11.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDecoratedMethod12() throws ClassHierarchyException, CancelException, IOException {
    // TODO: Change to 1, 1, 2 once https://github.com/wala/ML/issues/188 is fixed.
    test("tf2_test_decorated_method12.py", "f", 0, 0);
  }

  /**
   * Test https://github.com/wala/ML/issues/190.
   *
   * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#190 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testDecoratedMethod13() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_decorated_method13.py", "f", 1, 1, Map.of(2, Set.of(MNIST_INPUT)));
  }

  @Test
  public void testDecoratedFunctions()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_decorated_functions.py",
        "dummy_fun",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "dummy_test",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "test_function",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "test_function2",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "test_function3",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test(
        "tf2_test_decorated_functions.py",
        "test_function4",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a pytest with decorators. */
  @Test
  public void testDecoratedFunctions2()
      throws ClassHierarchyException, CancelException, IOException {
    test("test_decorated_functions.py", "test_dummy", 0, 0);
  }

  /**
   * Test a pytest without decorators that needs a PYTHONPATH. This is a "control" case. We'll add a
   * decorator in the next case.
   *
   * @see TestTensorflow2Model#testModule11().
   */
  @Test
  public void testDecoratedFunctions3()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {
          "proj48/src/__init__.py",
          "proj48/src/tf2_test_module9a.py",
          "proj48/src/tf2_test_module9b.py",
          "proj48/src/test_module10.py"
        },
        "src/tf2_test_module9b.py",
        "D.f",
        "proj48",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test a pytest without decorators. This is a "control." */
  @Test
  public void testDecoratedFunctions4()
      throws ClassHierarchyException, CancelException, IOException {
    test("test_decorated_functions2.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a pytest with a decorator. */
  @Test
  public void testDecoratedFunctions5()
      throws ClassHierarchyException, CancelException, IOException {
    test("test_decorated_functions3.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test a pytest with a decorator that needs a PYTHONPATH.
   *
   * @see TestTensorflow2Model#testModule11().
   */
  @Test
  public void testDecoratedFunctions6()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {
          "proj49/src/__init__.py",
          "proj49/src/tf2_test_module9a.py",
          "proj49/src/tf2_test_module9b.py",
          "proj49/src/test_module10.py"
        },
        "src/tf2_test_module9b.py",
        "D.f",
        "proj49",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /** Test a Pytest with a decorator without parameters. */
  @Test
  public void testDecoratedFunctions7()
      throws ClassHierarchyException, CancelException, IOException {
    test("test_decorated_functions4.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test a Pytest with a decorator without parameters that needs a PYTHONPATH.
   *
   * @see TestTensorflow2Model#testModule11().
   */
  @Test
  public void testDecoratedFunctions8()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {
          "proj50/src/__init__.py",
          "proj50/src/tf2_test_module10a.py",
          "proj50/src/tf2_test_module10b.py",
          "proj50/src/test_module11.py"
        },
        "src/tf2_test_module10b.py",
        "D.f",
        "proj50",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Test a Pytest with a decorator without parameters. The "test" is at the end of the filename.
   */
  @Test
  public void testDecoratedFunctions9()
      throws ClassHierarchyException, CancelException, IOException {
    test("decorated_function_test.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_reshape.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_6_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape2() throws ClassHierarchyException, CancelException, IOException {
    TensorType expectedType =
        new TensorType(
            FLOAT_32,
            asList(
                new SymbolicDim("?"), new NumericDim(28), new NumericDim(28), new NumericDim(1)));

    test("tf2_test_reshape2.py", "f", 1, 1, Map.of(2, Set.of(expectedType)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_reshape3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_6_INT32)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_reshape4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_28_28_1_FLOAT32)));
  }

  /** Test https://github.com/wala/ML/issues/195. */
  @Test
  public void testReshape5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_reshape5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_28_28_1_FLOAT32)));
  }

  /**
   * Regression guard for {@code tf.reshape(x, tf.shape(y))} shape inference. Runtime answer is
   * {@code (2, 3)} of {@code float32}. Post wala/ML#538's graceful-degradation fix in {@link
   * com.ibm.wala.cast.python.ml.client.Reshape} (mirroring {@link
   * com.ibm.wala.cast.python.ml.client.BroadcastTo}'s localized try/catch), the analysis no longer
   * throws on the {@code tf.shape(...)} shape arg. The inferred parameter type for {@code x} (vn=2)
   * is currently the imprecise {@code (⊤) of float32} (the opaque-shape-operand unknown-rank pin,
   * wala/ML#703) rather than the precise {@code (2, 3) float32}.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/473">wala/ML#473</a>): tighten the parameter
   * type to {@link #TENSOR_2_3_FLOAT32} when the helper learns to resolve {@code tf.shape(y)}'s
   * shape leaves precisely.
   */
  @Test
  public void testReshapeRuntimeShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_runtime_shape.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  @Test
  public void testConvertToTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Tier-1 generator (wala/ML#449): {@code tf.identity(input)} returns a fresh tensor with the same
   * shape and dtype as {@code input}. Pre-fix this routed via {@code identity}'s synthetic XML
   * which ultimately allocates a {@code Ltensorflow/python/framework/ops/convert_to_tensor} —
   * {@link com.ibm.wala.cast.python.ml.client.ConvertToTensor} handles dtype/shape, but the extra
   * indirection through {@code identity}'s wrapper class can be tightened.
   */
  @Test
  public void testIdentity()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_identity.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Companion to {@link #testIdentity()} exercising the keyword-argument call site {@code
   * tf.identity(input=...)}. The arg-resolution helpers in {@link
   * com.ibm.wala.cast.python.ml.client.Identity} (and the underlying {@code
   * getArgumentPointsToSet(builder, paramPos, paramName)}) resolve keyword args via {@code
   * paramName}; without a kwarg fixture that branch is dead-on-arrival in the test data.
   */
  @Test
  public void testIdentityKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_identity_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Tier-1 generator (wala/ML#449): {@code tf.stop_gradient(input)} returns a fresh tensor with the
   * same shape and dtype as {@code input}. Pre-fix this routed through {@code ReadDataFallback}
   * (the alloc has no value/dtype field bindings, just a {@code read_data} marker) and emitted
   * {@code [{? of unknown}]}; the {@link com.ibm.wala.cast.python.ml.client.StopGradient} generator
   * now reads {@code input}'s shape/dtype directly via the same {@code shapesOfArg} / {@code
   * dtypesOfArg} pattern as {@link com.ibm.wala.cast.python.ml.client.Sigmoid}.
   */
  @Test
  public void testStopGradient()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_stop_gradient.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /** Companion to {@link #testStopGradient()} exercising the keyword-argument call site. */
  @Test
  public void testStopGradientKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_stop_gradient_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Lock-in test for wala/ML#449 Tier-1 coverage of {@code tf.nn.bias_add(value, bias)}: returns a
   * fresh tensor with the same shape and dtype as {@code value} (bias is broadcast-added but
   * doesn't change the receiver's shape). Unlike {@link #testIdentity()} / {@link
   * #testStopGradient()} — which got dedicated {@link com.ibm.wala.cast.python.ml.client.Identity}
   * / {@link com.ibm.wala.cast.python.ml.client.StopGradient} generators paired with a direct
   * {@code <new>+<return>} in the XML — {@code bias_add} is modeled in {@code tensorflow.xml} as a
   * delegation: {@code <new>} of a {@code convert_to_tensor} function-object, {@code <call>} of its
   * {@code do} with {@code value} as the argument, then {@code <return>} of that result (no
   * dedicated Java generator; the actual tensor allocation happens inside {@code
   * convert_to_tensor.do()}). That delegation suffices for the shape/dtype-passthrough semantics
   * this test exercises.
   */
  @Test
  public void testBiasAdd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_bias_add.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testBiasAdd()} exercising the all-keyword call site {@code
   * tf.nn.bias_add(value=..., bias=...)}.
   */
  @Test
  public void testBiasAddKwarg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_bias_add_kwarg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Lock-in test for wala/ML#456: {@code tf.data.Dataset.reduce(initial_state, reduce_func)}
   * returns a tensor with the same shape/dtype as {@code initial_state}. Pre-fix the {@code
   * tensorflow/data/reduce.do()} XML called {@code read_data} virtually on its receiver, but the
   * {@code reduce} class didn't define {@code read_data} — the call was unresolved and {@code
   * def="xx"} bound to nothing. Pre/post-fix this test produces the same observable type ({@code
   * [{[] of int32}]}) for {@code f}'s parameter at {@code vn=2} because the initial-state's tensor
   * type still propagates via PA assignment edges; the test serves as a regression lock so a future
   * XML/PA refactor that breaks the propagation surfaces here instead of regressing silently. The
   * XML cleanup itself is hygiene — it removes the unresolved virtual call and aligns the model
   * with TF runtime semantics ({@code reduce(...)} returns {@code initial_state}'s shape/dtype).
   */
  @Test
  public void testDatasetReduce()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset_reduce.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testConvertToTensor2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  /**
   * Diagnostic for the existing {@code add_n} XML pattern (read list field 0 → {@code
   * convert_to_tensor}). Captures the observable static-analysis output for {@code tf.add_n([t1,
   * t2])} where both list elements are {@code tf.constant} tensors. If this test passes with {@code
   * TENSOR_3_INT32}, the list-element-PTS path through {@code <getfield class="Llist" field="0">}
   * works and Tier 5 ops ({@code concat}/{@code stack}/{@code meshgrid}) can use the same pattern
   * cheaply. If it produces {@code ? of unknown}, the pattern doesn't propagate types and Tier 5
   * needs Java-side list-element traversal logic.
   */
  @Test
  public void testAddN()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add_n.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Tier-5 generator (wala/ML#449): {@code tf.concat(values, axis)}. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Concat} generator computes the precise output shape by
   * walking every entry in the {@code values} list, summing each input's dim along the resolved
   * {@code axis}, and inheriting the rest of the shape from the first input. The fixture
   * concatenates two {@code (3,)} tensors along {@code axis=0}, so the precise output is {@code
   * (6,)}; dtype is inherited from the first element ({@code int32}).
   */
  @Test
  public void testConcat()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_concat.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_6_INT32)));
  }

  /**
   * Multi-rank {@code tf.concat([t1, t2], axis=1)} with {@code (2, 3)} inputs. Exercises the
   * rank-aware path in {@link com.ibm.wala.cast.python.ml.client.Concat#computeConcatenatedShape}:
   * non-axis dim preservation (the leading {@code 2} survives) and the axis-dim sum (the trailing
   * {@code 3 + 3 = 6}).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testConcatMultirank()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_concat_multirank.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_6_INT32)));
  }

  /**
   * {@code tf.concat([t1, t2], axis=-1)} with {@code (2, 3)} inputs. Exercises the negative-axis
   * normalization in {@link com.ibm.wala.cast.python.ml.client.Concat#computeConcatenatedShape}:
   * {@code axis = -1} resolves to {@code rank - 1 = 1} for rank-2 inputs, producing the same {@code
   * (2, 6)} answer as the explicit {@code axis=1} fixture.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testConcatNegativeAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_concat_negaxis.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_6_INT32)));
  }

  /**
   * Tier-5 generator (wala/ML#449): {@code tf.stack(values, axis)}. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Stack} generator computes the precise output shape by
   * reading the {@code values} list's PTS-derived length {@code N} and inserting it at the resolved
   * {@code axis} position into the first element's shape: {@code values[0].shape[:axis] + (N,) +
   * values[0].shape[axis:]}. The fixture stacks two {@code (3,)} tensors with {@code axis=0}, so
   * the precise output is {@code (2, 3)}; dtype is inherited from the first element ({@code
   * int32}).
   */
  @Test
  public void testStack()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_stack.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
  }

  /**
   * Generator test for {@code tf.einsum(equation, *inputs)}. The {@link
   * com.ibm.wala.cast.python.ml.client.Einsum} generator parses the equation string and composes
   * the output shape from each input's shape. Here {@code einsum("ij,jk->ik", a, b)} with {@code a}
   * and {@code b} both {@code (2, 2)} yields {@code (2, 2)}. Output dtype inherits from the first
   * tensor input ({@code float32}). See wala/ML#507.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_einsum.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Implicit-output einsum: with no {@code ->}, the output labels are those occurring exactly once,
   * in alphabetical order, so {@code "ij,jk"} composes the same {@code (2, 2)} shape as {@code
   * "ij,jk->ik"}. See wala/ML#507.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumImplicit()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_einsum_implicit.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Broadcasting-ellipsis einsum: the parser doesn't model {@code ...}, so it soundly falls back to
   * an unknown ({@code ⊤}) shape while keeping the dtype precise. The Python runtime shape is
   * {@code (2, 2)}; the analysis reports {@code ⊤} until the ellipsis form is modeled.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/705">wala/ML#705</a>): tighten to the
   * precise shape once ellipsis and diagonal einsum forms are handled.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumEllipsisFallback()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_einsum_ellipsis.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Diagonal einsum (a repeated label within one term, {@code "ii->i"}): the parser doesn't model
   * diagonal extraction, so it soundly falls back to an unknown ({@code ⊤}) shape while keeping the
   * dtype precise. The Python runtime shape is {@code (2,)}; the analysis reports {@code ⊤} until
   * the diagonal form is modeled.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/705">wala/ML#705</a>): tighten to the
   * precise shape once ellipsis and diagonal einsum forms are handled.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumDiagonalFallback()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_einsum_diag.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Shape-vector provenance (wala/ML#703): {@code tf.reshape(x, t.shape.as_list()[-2:])} resolves
   * to the source tensor's trailing sub-shape. The shape argument's points-to set is empty (the
   * {@code shape} member, {@code as_list}, and the slice are unmodeled in the heap), so {@code
   * Reshape} recovers it by def-use provenance: the slice of the {@code as_list()} of the {@code
   * .shape} of {@code t} of shape {@code (4, 5, 6)}, sliced with {@code [-2:]}, is {@code (5, 6)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeAsListSlice()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_as_list_slice.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_6_FLOAT32)));
  }

  /**
   * Unsliced companion of {@link #testShapeAsListSlice()} (wala/ML#703): {@code tf.reshape(x,
   * t.shape.as_list())} resolves to the source tensor's full shape.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeAsList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_as_list.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_5_6_FLOAT32)));
  }

  /**
   * Variable-bound companion of {@link #testShapeAsListSlice()} (wala/ML#703, wala/ML#704): the
   * slice bound is a negated local ({@code shape[-k:]} with a constant {@code k}), mirroring
   * NLPGNN's {@code einsum_via_matmul} idiom ({@code input_shape[-num_inner_dims:]}). The unary
   * negation is constant-folded by the slice-bound resolver, so the trailing sub-shape resolves to
   * the precise {@code (5, 6)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeAsListSliceVar()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_as_list_slice_var.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_6_FLOAT32)));
  }

  /**
   * Interprocedural shape-vector provenance (wala/ML#706): the shape list is produced by a user
   * helper ({@code def get_shape(t): return t.shape.as_list()}, the BERT/ALBERT {@code
   * get_shape_list} pattern), so the def-use walk follows the helper invoke to its returned {@code
   * .shape.as_list()} chain; the callee parameter's interprocedural points-to set resolves the
   * source tensor. The {@code [-2:]} slice of {@code (4, 5, 6)} is the precise {@code (5, 6)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeHelperSlice()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_helper_slice.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_6_FLOAT32)));
  }

  /**
   * Combines {@link #testShapeHelperSlice()}'s interprocedural hop with {@link
   * #testShapeAsListSliceVar()}'s negated variable bound: {@code get_shape(t)[-k:]} with a constant
   * {@code k} is structurally NLPGNN's {@code einsum_via_matmul} shape read ({@code
   * get_shape_list(input_tensor)[-num_inner_dims:]}). See wala/ML#706 and wala/ML#704.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeHelperSliceVar()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_helper_slice_var.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_6_FLOAT32)));
  }

  /**
   * {@code np.prod} over a shape-derived list (wala/ML#707): {@code [np.prod(get_shape(t)[-2:])]}
   * folds the product of the static trailing dimensions ({@code 5 * 6}) into the reshape target, so
   * the result is the precise {@code (30,)}. Mirrors NLPGNN's {@code einsum_via_matmul} ({@code
   * inner_dim = np.prod(input_shape[-num_inner_dims:])}).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeProd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_prod.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_FLOAT32)));
  }

  /**
   * NLPGNN's {@code DenseLayer3d} einsum path in miniature (wala/ML#704): the weight is built flat
   * from configuration fields, reshaped to rank 3 in {@code call} (the {@code hidden} leading dim
   * stays dynamic since {@code build}'s {@code input_shape} subscript doesn't resolve), and
   * consumed by {@code einsum("BFH,HND->BFND", ...)}. The parser refines the contracted {@code H}
   * label's dynamic occurrence with the input's known {@code 6} and composes the precise runtime
   * {@code (2, 4, 3, 5)}: the trailing head dims are static, per the issue's expectation.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dEinsum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_einsum.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4, 3, 5))));
  }

  /**
   * Operand-order companion of {@link #testDense3dEinsum()} (wala/ML#704): the weight, whose
   * contracted dim is dynamic, comes first in the equation ({@code "HND,BFH->BFND"}), so the
   * input's statically-known occurrence of the shared {@code H} label arrives second and refines
   * the earlier dynamic one. The composed shape is the same precise {@code (2, 4, 3, 5)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testDense3dEinsum2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dense3d_einsum2.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4, 3, 5))));
  }

  /**
   * NLPGNN's {@code einsum_via_matmul} matmul path in miniature, end to end (wala/ML#704): the
   * {@code get_shape_list} hop, the negated-parameter slice bounds, the {@code np.prod} folds, and
   * the {@code batch_dims + outer_dims} concatenation (wala/ML#708) all resolve, so the reshape arm
   * types as the precise runtime {@code (2, 4, 3, 5)}. The {@code (4, 5)} member is the other arm
   * of the {@code len(outer_dims) > 1} guard's φ (the raw rank-2 matmul), which flows statically
   * even though the runtime always takes the reshape arm here.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumViaMatmul()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_via_matmul.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4, 3, 5), TensorType.of(FLOAT_32, 4, 5))));
  }

  /**
   * The two-inner-dims variant of {@link #testEinsumViaMatmul()} (NLPGNN's {@code DenseLayer3dProj}
   * shape, {@code einsum_via_matmul(input_tensor, w, 2)}): exercises the {@code batch_dims +
   * [inner_dim]} concatenation of a shape vector with a literal list whose element is an {@code
   * np.prod} fold (wala/ML#708). The reshape-then-matmul arm types as the precise runtime {@code
   * (2, 4, 6)}; the {@code (3, 6)} member is the untaken arm of the {@code num_inner_dims > 1}
   * guard's φ (the rank-2 matmul of the unreshaped input), which flows statically.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEinsumViaMatmul2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_einsum_via_matmul2.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4, 6), TensorType.of(FLOAT_32, 3, 6))));
  }

  /**
   * Guard companion of {@link #testShapeProd()} (wala/ML#707): {@code np.prod} with an extra
   * argument ({@code axis=0}) can change the result's rank, so the fold refuses it and the shape
   * position degrades to a dynamic dimension in the walk-side contexts. The interpreter path
   * evaluates the call concretely and contributes the precise {@code (30,)} in its context, so the
   * union carries both.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeProdAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_shape_prod_axis.py",
        "f",
        1,
        1,
        Map.of(
            2, Set.of(TENSOR_30_FLOAT32, new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE)))));
  }

  /**
   * Concatenation of two shape vectors (wala/ML#708): {@code get_shape(t)[:1] + get_shape(t)[-2:]}
   * over {@code (4, 5, 6)} composes the reshape target {@code (4, 5, 6)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeConcat()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_shape_concat.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_5_6_FLOAT32)));
  }

  /**
   * Guard companion of {@link #testShapeAsListSlice()} (wala/ML#703): a non-unit step over a shape
   * list ({@code [::2]}) is unmodeled, so the walk soundly reports an unknown ({@code ⊤}) shape
   * while keeping the dtype precise. The Python runtime shape is {@code (4, 6)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeSliceStep()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_shape_slice_step.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Guard companion of {@link #testShapeAsListSlice()} (wala/ML#703): the slice bound is a φ of two
   * constants, so within one context its points-to set is ambiguous and the bound resolver must not
   * assert either slicing; the walk soundly reports an unknown ({@code ⊤}) shape. A bound that is a
   * distinct constant per calling context stays precise (context sensitivity disambiguates it); the
   * φ forces the ambiguity into a single context.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeSliceAmbiguous()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_shape_slice_ambiguous.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Tier-5 generator (wala/ML#449): {@code tf.math.top_k(input, k)}. Returns a {@code (values,
   * indices)} 2-tuple. The dedicated {@link com.ibm.wala.cast.python.ml.client.TopK} generator
   * implements {@link com.ibm.wala.cast.python.ml.client.TupleElementProvider} with per-index dtype
   * precision ({@code values} inherits input dtype; {@code indices} is fixed at {@code int32}), but
   * the wrap-on-property-read dispatch in {@link
   * com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory} doesn't fire for NamedTuple-style
   * attribute access ({@code result.values} / {@code result.indices}). Until <a
   * href="https://github.com/wala/ML/issues/480">wala/ML#480</a> is fixed, both destructured
   * elements receive the aggregate union of per-element types.
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/480">wala/ML#480</a> lands, narrow the
   * assertion to {@code Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)} (precise {@code values} dtype).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTopKValues()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_top_k.py",
        "f_values",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32, TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * Counterpart of {@link #testTopKValues()} for the {@code indices} element of {@code
   * tf.math.top_k}'s tuple result. Same wala/ML#480-driven imprecision: the assertion captures the
   * observed aggregate union with a TODO to narrow once the per-index dispatch is fixed.
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/480">wala/ML#480</a> lands, narrow the
   * assertion to {@code Set.of(TENSOR_INT32_UNKNOWN_SHAPE)} (precise {@code indices} dtype).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTopKIndices()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_top_k.py",
        "f_indices",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32, TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * Tier-5 generator (wala/ML#449): {@code tf.meshgrid(*xi)}. Returns N tensors (one per input)
   * sharing the broadcast of input shapes and the first input's dtype. The dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Meshgrid} generator implements {@link
   * com.ibm.wala.cast.python.ml.client.TupleElementProvider}, but the XML only allocates one tuple
   * slot (field 0); the second meshgrid output ({@code Y} in the fixture) doesn't have a backing
   * alloc, so it falls through to ⊤/UNKNOWN and the aggregate union leaks the {@code
   * float32}/{@code unknown} pair. See <a href="https://github.com/wala/ML/issues/480">
   * wala/ML#480</a>.
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/480">wala/ML#480</a> lands (or the
   * meshgrid XML is updated to allocate per-input tuple slots), narrow the assertion to {@code
   * Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testMeshgrid()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_meshgrid.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32, TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Companion to {@link #testMeshgrid}: 2-parameter sink {@code f(X, Y)} so the analyzer's
   * per-parameter typing on `tf.meshgrid`'s tuple result can be observed at distinct value numbers.
   * The two parameters split the leak asymmetrically:
   *
   * <ul>
   *   <li>vn=2 ({@code X}) shows the full {@code float32}/⊤-dtype union &mdash; the meshgrid XML
   *       only allocates field-0 of the tuple, so when {@code X} aliases that slot through PA
   *       propagation it picks up both the precise float32 alloc and the ⊤ tuple-slot leak.
   *   <li>vn=3 ({@code Y}) collapses to just the precise {@code float32} &mdash; the second
   *       meshgrid output's allocation site doesn't reach this parameter through the PA graph, so
   *       the ⊤ leak doesn't surface.
   * </ul>
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/480">wala/ML#480</a> lands, narrow
   * vn=2's assertion to {@code Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)} (vn=3 is already there).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testMeshgridXY()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_meshgrid_xy.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32, TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE),
            3, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Generator-dispatch test for the 3-arg form of {@code tf.where(condition, x, y)}. The dedicated
   * {@link com.ibm.wala.cast.python.ml.client.Where} generator produces shape and dtype by unioning
   * the inferred sets over {@code x} and {@code y} (and intentionally ignoring {@code condition}'s
   * shape, per the modeling note in {@code Where}'s class Javadoc). The fixture has all three
   * operands shape {@code (3,)} float32, so the union collapses to the singleton {@code (3,)
   * float32}. Fix for the wala/ML#422 listing.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testWhere()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_where.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Sibling of {@link #testWhere()} that exercises the union-over-{@code x}-and-{@code y} path with
   * broadcast-compatible different shapes: {@code x} is {@code (3,)} float32 and {@code y} is
   * {@code (2, 3)} float32. The runtime broadcast result is {@code (2, 3)}; the static analysis
   * unions the two operand shapes to {@code {(3,), (2, 3)}}, which is sound but imprecise.
   *
   * <p>TODO(<a href="https://github.com/wala/ML/issues/482">wala/ML#482</a>): once broadcast-shape
   * composition lands, tighten this assertion from the union {@code {TENSOR_3_FLOAT32,
   * TENSOR_2_3_FLOAT32}} to the precise {@code TENSOR_2_3_FLOAT32}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testWhereBroadcast()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_where_broadcast.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_FLOAT32, TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testConvertToTensor4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testConvertToTensor5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testConvertToTensor6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor8.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testConvertToTensor9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor9.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor10.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testConvertToTensor11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor11.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testConvertToTensor12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor12.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
  }

  @Test
  public void testOneHot()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot8.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot9.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot10.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot11.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot12.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot13.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot14()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot14.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  @Test
  public void testOneHot15()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot15.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  @Test
  public void testOneHot16()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot16.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  @Test
  public void testOneHot17()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot17.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testOneHot18()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot18.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_5_3_FLOAT32)));
  }

  @Test
  public void testOneHot19()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // Fixed by handling `CONSTANT_OP_CONSTANT`.
    test("tf2_test_one_hot19.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
    test("tf2_test_one_hot19.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_5_3_FLOAT32)));
  }

  @Test
  public void testOneHot20()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_one_hot20.py",
        "test",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_3_3_FLOAT32),
            3, Set.of(TENSOR_3_3_INT32),
            4, Set.of(TENSOR_3_3_FLOAT32),
            5, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testEye()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testEye2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testEye3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testEye4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_2_FLOAT32)));
  }

  @Test
  public void testEye5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_2_3_FLOAT32)));
  }

  @Test
  public void testEye6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_2_3_FLOAT32)));
  }

  @Test
  public void testGamma()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testGamma2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT64)));
  }

  @Test
  public void testGamma3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_7_5_2_FLOAT32)));
  }

  @Test
  public void testGamma4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testGamma5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testGamma6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testGammaMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testFillKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_fill_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
  }

  @Test
  public void testFillMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_fill_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testRangeStartLimitKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_start_limit_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_INT32)));
  }

  @Test
  public void testRange1PosLimitDeltaKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_1_pos_limit_delta_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testRange1PosDeltaKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {

    test("tf2_test_range_1_pos_delta_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_INT32)));
  }

  @Test
  public void testRangeStartDeltaKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_start_delta_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testRangeStartKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_start_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testRangeKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testRangeMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testPoisson()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testPoisson2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testPoisson3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT64)));
  }

  @Test
  public void testPoisson4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_7_5_2_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.sparse.from_dense(tensor, name=None)}. The fixture uses
   * the keyword form {@code from_dense(tensor=x)} so that arg-resolution drives through {@link
   * com.ibm.wala.cast.python.ml.client.SparseFromDense}'s {@code Parameters.TENSOR.getName()}
   * keyword-name lookup, exercising the {@code Locale.ROOT} line that wala/ML#510 flagged as
   * uncovered. Output shape and dtype both inherit from {@code tensor}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSparseFromDense()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_from_dense.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseEye() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_5_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseEye2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_5_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseEye3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_5_INT32.asSparse())));
  }

  @Test
  public void testSparseEye4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_2_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseEye5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_2_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseEye6() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_2_INT32.asSparse())));
  }

  @Test
  public void testRaggedConstant() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedConstant2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedConstant3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_INT32)));
  }

  @Test
  public void testRaggedConstant4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedConstant5() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant5.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedConstant6() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testRaggedConstant7() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedConstant8() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant8.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedConstant9() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant9.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedConstant10() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant10.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_1_FLOAT32)));
  }

  @Test
  public void testRaggedConstant11() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant11.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedConstant12() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant12.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_2_FLOAT32)));
  }

  @Test
  public void testRaggedConstant13() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant13.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_0_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedConstant14() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant14.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_0_RAGGED_3_FLOAT32)));
  }

  @Test
  public void testRaggedConstant15() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant15.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_2_INT32)));
  }

  @Test
  public void testRaggedConstant16() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant16.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_2_INT32)));
  }

  /**
   * Test non-uniform inner dimensions.
   *
   * <p>TODO: Remove expected assertion error once https://github.com/wala/ML/issues/350 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testRaggedConstant17() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant17.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_2_3_INT32)));
  }

  /** This one works because the inner dimensions are uniform. */
  @Test
  public void testRaggedConstant18() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant18.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_2_2_INT32)));
  }

  @Test
  public void testRaggedConstantKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant_keyword.py",
        "test",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_2_RAGGED_INT32),
            3, Set.of(TENSOR_2_RAGGED_INT32),
            4, Set.of(TENSOR_2_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromRowSplits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_splits.py",
        "test_ragged_from_row_splits",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_5_RAGGED_INT32),
            3, Set.of(TENSOR_5_RAGGED_INT32),
            4, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testRaggedRange() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_10_INT32)));
  }

  @Test
  public void testRaggedRange2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_10_INT32)));
  }

  @Test
  public void testRaggedRange3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedRange4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedRange5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedRange6() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedRange7() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  /**
   * Canonical case for <a href="https://github.com/wala/ML/issues/546">wala/ML#546</a>: {@code
   * tf.ragged.range(3, 18, 3)} — all three scalar args are compile-time literals, so the inner
   * length is statically computable: {@code ceil((18 - 3) / 3) = 5}. The analyzer pins {@code (1,
   * 5)} instead of {@code (1, ragged)}.
   */
  @Test
  public void testRaggedRange8() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range8.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_5_INT32)));
  }

  /**
   * Multi-length fallback case for <a href="https://github.com/wala/ML/issues/546">wala/ML#546</a>:
   * {@code start} resolves to two literal values via an if/else, so the cross-product yields
   * lengths {@code {10, 8}}. {@code computeStaticInnerLength} returns {@code null} and the inner
   * dim falls back to {@code RaggedDim}.
   */
  @Test
  public void testRaggedRange9() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range9.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_RAGGED_INT32)));
  }

  @Test
  public void testRaggedRangeKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_range_keyword.py",
        "test",
        5,
        5,
        Map.of(
            2, Set.of(TENSOR_1_5_INT32),
            3, Set.of(TENSOR_1_5_INT32),
            4, Set.of(TENSOR_1_5_INT32),
            5, Set.of(TENSOR_1_5_INT32),
            6, Set.of(TENSOR_1_5_INT32)));
  }

  @Test
  public void testSparseTensor() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_tensor.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_4_INT32.asSparse())));
  }

  @Test
  public void testInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_input.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_NONE_32_FLOAT32)));
  }

  @Test
  public void testInput2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_input2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_NONE_NONE_STRING)));
  }

  /**
   * Regression guard for wala/ML#355: when the {@code shape} argument to {@code tf.keras.Input} is
   * unresolvable from the static analyzer's perspective (here, sourced from {@code json.loads},
   * which Ariadne does not model), {@code Input.getDefaultShapes} must return {@code null} (⊤,
   * tensor with unknown shape) rather than {@code Collections.emptySet()} (⊥, not a tensor). The ⊥
   * return previously made the call's result silently disappear from the tensor analysis despite
   * being a tensor at runtime; the fix in ponder-lab/ML@078208f6 restores ⊤ propagation, so the
   * call site is recognized as a tensor with concrete dtype but unknown shape.
   */
  @Test
  public void testInputUnresolvableShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_input_unresolvable_shape.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/604">wala/ML#604</a>: when an
   * allocator's {@code shape} argument is unresolvable (here {@code tf.zeros(json.loads(...))},
   * whose source Ariadne does not model), {@code TensorTypeAllocator.getDefaultShapes} must return
   * {@code null} (⊤, tensor with unknown shape) rather than throwing {@code
   * UnsupportedOperationException}, which previously aborted the whole analysis. Recovering the
   * content-dependent shape itself is the user-annotation problem tracked by wala/ML#370; this
   * guard only pins the non-crashing ⊤ floor.
   */
  @Test
  public void testZerosUnresolvableShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_zeros_unresolvable_shape.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  @Test
  public void testRaggedFromNestedRowLengths()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_lengths.py",
        "test_ragged_from_nested_row_lengths",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowLengthsKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_lengths_keyword.py",
        "test_ragged_from_nested_row_lengths_keyword",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowLengthsKeyword2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_lengths_keyword2.py",
        "test_ragged_from_nested_row_lengths_keyword2",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowSplitsPositional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_splits_positional.py",
        "test_ragged_from_nested_row_splits_positional",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowSplitsKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_splits_keyword.py",
        "test_ragged_from_nested_row_splits_keyword",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowSplitsMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_splits_mixed.py",
        "test_ragged_from_nested_row_splits_mixed",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsPositional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_positional.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_STRING)));
  }

  @Test
  public void testRaggedNestedValueRowidsPositionalLists()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_positional_lists.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_STRING)));
  }

  @Test
  public void testRaggedNestedValueRowidsPositionalTuples()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_positional_tuples.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_STRING)));
  }

  @Test
  public void testRaggedNestedValueRowidsKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_keyword.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsKeywordLists()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_keyword_lists.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsKeywordTuples()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_keyword_tuples.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_mixed.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_RAGGED_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsMixedLists()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_mixed_lists.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_RAGGED_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsMixedTuples()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_mixed_tuples.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_RAGGED_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedFromNestedValueRowIdsComplete()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // check_case_1: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_1",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));

    // check_case_2: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_2",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));

    // check_case_3: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_3",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));

    // check_case_4: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_4",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));

    // check_case_5: [2, None, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_5",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));

    // check_case_6: [2, None], float32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_6",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_FLOAT32)));

    // check_case_7: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_7",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));
  }

  @Test
  public void testVariablePositional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_variable_positional.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testVariableKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_variable_keyword.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
    test("tf2_test_variable_keyword.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testVariableMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_variable_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT64)));
  }

  @Test
  public void testVariablePositionalComplex()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {

    test("tf2_test_variable_positional_complex.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testVariablePositionalShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {

    test(
        "tf2_test_variable_positional_shape.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  @Test
  public void testVariablePositionalDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {

    test("tf2_test_variable_positional_dtype.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT64)));
  }

  @Test
  public void testModelInit()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_model_init.py", "check_positional", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test("tf2_test_model_init.py", "check_keyword", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test("tf2_test_model_init.py", "check_mixed", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test("tf2_test_model_init.py", "check_subclass", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
    test(
        "tf2_test_model_init.py",
        "check_multiple",
        2,
        2,
        Map.of(2, Set.of(TENSOR_1_2_FLOAT32), 3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Pins the output type of a functional {@code tf.keras.Model} call whose output shape <em>differs
   * from</em> its input shape: a {@code Dense(3)} model maps a {@code (1, 2)} call input to a
   * {@code (1, 3)} output. {@code ModelCall} recovers the model's output generator (the {@code
   * Dense(3)} call, reached via the {@code outputs} construction argument) and reports the
   * transformed {@code (1, 3)} shape — both for positional ({@code Model(in, out)}) and keyword
   * ({@code Model(outputs=...)}) construction. Before wala/ML#537, {@code ModelCall} fell back to
   * the call's input shape when the output generator wasn't reached, which would have reported the
   * unsound {@code (1, 2)} here (input shape, not output). Companion to {@link #testModelInit}
   * (whose {@code Dense(2)}-on-dim-2 models are shape-preserving, so they can't distinguish the two
   * behaviors) and to {@link #testGanTutorialGeneratorLoss} (whose deep convolutional chain leaves
   * the output generator shapeless, exercising the ⊤ path).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testModelCallOutputShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call_output_shape.py",
        "consume_positional",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_3_FLOAT32)));
    test(
        "tf2_test_model_call_output_shape.py",
        "consume_keyword",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_3_FLOAT32)));
  }

  @Test
  public void testConvertToTensor13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_convert_to_tensor13.py",
        "test",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_3_FLOAT32),
            3, Set.of(TENSOR_2_2_INT32),
            4, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testEye7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_eye7.py",
        "test",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_2_3_INT32),
            3, Set.of(TENSOR_3_3_FLOAT32),
            4, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testGamma7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gamma7.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_FLOAT64),
            3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testPoisson5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_poisson5.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_FLOAT64),
            3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testSparseAdd7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_add7.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_2_INT32.asSparse()),
            3, Set.of(TENSOR_2_2_INT32.asSparse())));
  }

  @Test
  public void testSparseEye7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_eye7.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_3_INT32.asSparse()),
            3, Set.of(TENSOR_3_3_FLOAT32.asSparse())));
  }

  @Test
  public void testDense()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));
  }

  /**
   * Test for <a href="https://github.com/wala/ML/issues/371">wala/ML#371</a>. A single {@code
   * Dense} layer call inside {@code M.call} with a {@code tf.keras.Input} parameter.
   *
   * <p>Two tensor variables are found: the {@code x} parameter (v3, shape {@code (None, 3)}) and
   * the {@code Dense} result (v25, shape {@code (None, 4)}). Both are correct source-level tensors
   * under a single trampoline context.
   */
  @Test
  public void testDenseModelCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense_model_call.py", "M.call", 1, 2, Map.of(3, Set.of(TENSOR_NONE_3_FLOAT32)));
  }

  @Test
  public void testDense2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense2.py", "consume1", 1, 1, Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));
    test("tf2_test_dense2.py", "consume2", 1, 1, Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  /**
   * Chained {@code Dense} layer calls at module level, where the second layer's {@code inputs}
   * argument is the return value of the first layer's call. Exercises shape propagation through a
   * layer-call result at script-body scope.
   */
  @Test
  public void testDenseChain()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense_chain.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  /**
   * Chained {@code Dense} layer calls inside a {@code tf.keras.Model.__call__} method body with
   * direct {@code self.layer1} / {@code self.layer2} attribute reads. Exercises shape propagation
   * through a layer-call result inside a user-defined class method.
   */
  @Test
  public void testDenseChain2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense_chain2.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  /**
   * Chained {@code Dense} layer calls inside a {@code tf.keras.Model.__call__} method body, where
   * the layers are iterated via a {@code for} loop over a {@code self.layers_list} attribute rather
   * than being accessed by direct attribute name. Exercises shape propagation through a loop-phi'd
   * local whose points-to set spans every list element.
   */
  @Test
  public void testDenseChain3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dense_chain3.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));
  }

  /**
   * Exercise the duck-typed {@code numpy.ndarray.astype(...)} dispatch path added for wala/ML#356.
   * The receiver is {@code x_train}, the first element of {@code
   * tf.keras.datasets.mnist.load_data()}. Without mnist modeling, the receiver's concrete shape
   * cannot be resolved; after the {@code astype} call, {@code consume}'s parameter is recognised as
   * a float32 tensor with a {@code null} dims list.
   */
  @Test
  public void testAstype()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_astype.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_60000_28_28_FLOAT32)));
  }

  /**
   * Regression guard for wala/ML#403: chained {@code x.astype(int32).astype(float32)} on an mnist
   * receiver. The first cast's result is a synthetic-method return whose PointerKey is implicit, so
   * the receiver-shape lookup for the second {@code astype} call hits the {@code
   * IllegalArgumentException}-catch fallback path in {@link AstypeOperation#getDefaultShapes},
   * which returns {@code null} (⊤) for shape while dtype still resolves to {@code float32}.
   *
   * <p>The runtime-vs-analyzer asymmetry here ({@code (60000, 28, 28) float32} at runtime vs.
   * {@code TensorType(float32, null)} from the analyzer) is the same kind of deliberate limitation
   * as {@link #testInputUnresolvableShape}: traversing implicit-PK chains across synthetic-method
   * returns is a known architectural gap (wala/ML#402 / wala/WALA#1889), and returning ⊤ rather
   * than ⊥ is the lattice-correct response in the meantime — dtype still carries through, so
   * downstream analysis isn't dropped. The test asserts the analyzer's lattice-correct output
   * rather than suppressing it as a would-be-fixed failure; if and when the implicit-PK chain
   * traversal lands, this expectation flips to {@code TENSOR_60000_28_28_FLOAT32} as part of that
   * change.
   */
  @Test
  public void testAstypeChained()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_astype_chained.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Exercise {@link NdarrayReshape#getDefaultShapes}'s SSA-substrate DU walk: resolves the {@code
   * -1} in {@code x_train.reshape([-1, 784])} by tracing the receiver back to {@code mnist.x_train}
   * ({@code (60000, 28, 28)}). Guards {@link TensorGenerator#getShapesOrSSAChain} against
   * regression.
   *
   * <p>Dtype propagation ({@code uint8}) uses the existing {@code getDTypes(builder, receiverVn)}
   * path, which resolves through the normal PA because {@code x_train}'s dtype is carried on the
   * {@link MnistInputData}-manufactured allocation.
   */
  @Test
  public void testNdarrayReshape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ndarray_reshape.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_60000_784_UINT8)));
  }

  /**
   * Guards function-style {@code np.array(x, dtype)} shape-preservation. Mirrors {@link
   * #testAstype}'s shape/dtype contract but through the function-call path rather than the
   * method-call path: {@code np.array(x_train, np.float32)} should yield a tensor with {@code
   * x_train}'s shape {@code (60000, 28, 28)} and dtype {@code float32}.
   *
   * <p>Positive regression guard for wala/ML#404: {@code np.array} is modeled in {@code numpy.xml}
   * as a {@code Lnumpy/array} class whose {@code do} method returns a fresh {@code Lnumpy/ndarray},
   * and {@link NpArray} reads shape from arg 0 ({@code x}) and dtype from arg 1 ({@code dtype}).
   */
  @Test
  public void testNpArrayPreservesShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_np_array_preserves_shape.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_60000_28_28_FLOAT32)));
  }

  /**
   * Pins {@link NpOnes} directly (isolated from the {@code tf.constant} bridge of {@link
   * #testConstantFromNumpy}): {@code consume_ones(x)} where {@code x = np.ones((2, 3),
   * dtype=np.float32)} should yield {@code (2, 3) float32} &mdash; shape from the shape-tuple
   * argument, dtype from the explicit {@code dtype} argument. Positive regression guard for
   * wala/ML#539.
   */
  @Test
  public void testNpOnes()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_np_ones_zeros.py", "consume_ones", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins {@link NpZeros} directly: {@code consume_zeros(x)} where {@code x = np.zeros((4,),
   * dtype=np.int64)} should yield {@code (4,) int64}. Companion to {@link #testNpOnes}; positive
   * regression guard for wala/ML#539.
   */
  @Test
  public void testNpZeros()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_np_ones_zeros.py", "consume_zeros", 1, 1, Map.of(2, Set.of(TENSOR_4_INT64)));
  }

  /**
   * Pins {@link NpOnes}'s default dtype: {@code consume_ones_default(x)} where {@code x =
   * np.ones((2, 3))} (no {@code dtype} argument) should yield {@code (2, 3) float64}, since NumPy
   * defaults to {@code float64} (unlike {@code tf.ones}, which defaults to {@code float32}). Guards
   * the {@code float64} default override in {@link NpOnes#getDefaultDTypes} for wala/ML#539.
   */
  @Test
  public void testNpOnesDefaultDtype()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_np_ones_zeros.py",
        "consume_ones_default",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT64)));
  }

  /**
   * Guards slice-receiver dtype recovery through a chained slice (<a
   * href="https://github.com/wala/ML/issues/602">wala/ML#602</a>): {@code x_train[:5][:3]} on a
   * {@code (60000, 28, 28) uint8} ndarray yields a {@code (3, 28, 28) uint8} tensor. The outer
   * slice's receiver is the inner slice's result, whose dtype the PTS walk can't see; {@link
   * com.ibm.wala.cast.python.ml.client.TensorGenerator#dtypesFromSSAChain} recovers it by recursing
   * through the dtype-preserving slice op rather than falling back to {@code DType.UNKNOWN}.
   */
  @Test
  public void testSliceChainedDtype()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_slice_chained_dtype.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_28_28_UINT8)));
  }

  /**
   * Guards slice-receiver dtype recovery through a {@code reshape(pad(x))} chain (<a
   * href="https://github.com/wala/ML/issues/602">wala/ML#602</a>), the MRE distilled from
   * MusicTransformer's {@code RelativeGlobalAttention._skewing}. The slice receiver is {@code
   * tf.reshape} of {@code tf.pad}; neither op is itself dtype-modeled ({@code tf.pad} is unmodeled
   * entirely), so the receiver dtype lands at ⊤ unless {@link
   * com.ibm.wala.cast.python.ml.client.TensorGenerator#dtypesFromSSAChain} recurses through the
   * dtype-preserving chain to the concrete {@code float32} input.
   *
   * <p>TODO: the shape stays {@code (1, 1, 3, 2)} (the reshape result) rather than {@code (1, 1, 2,
   * 2)} &mdash; the {@code [:, :, 1:, :]} subscript isn't reducing axis 2, a shape-precision gap
   * tracked by <a href="https://github.com/wala/ML/issues/607">wala/ML#607</a>. This test pins the
   * dtype recovery and the currently-observed shape.
   */
  @Test
  public void testSliceReshapePadDtype()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_slice_reshape_pad_dtype.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_1_3_2_FLOAT32)));
  }

  /**
   * Guards the {@code k}-default path of the top_k composer (<a
   * href="https://github.com/wala/ML/issues/609">wala/ML#609</a>): with {@code k} omitted it
   * defaults to {@code 1}, so {@code values} of a {@code (4,)} input is {@code (1,)} float32.
   */
  @Test
  public void testTopkDefaultK()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_topk_default_k.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_1_FLOAT32)));
  }

  /**
   * Guards the non-constant-{@code k} path of the top_k composer (<a
   * href="https://github.com/wala/ML/issues/609">wala/ML#609</a>): when {@code k} is not a
   * resolvable integer constant (here from {@code json.loads}), the shape can't be composed and
   * degrades to ⊤ rather than guessing. The dtype stays precise (float32).
   */
  @Test
  public void testTopkNonConstantK()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_topk_nonconstant_k.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Guards the unknown-input-shape path of the top_k composer (<a
   * href="https://github.com/wala/ML/issues/609">wala/ML#609</a>): when the input tensor's shape is
   * ⊤ (here {@code tf.ones(json.loads(...))}), {@code input.shape[:-1] + (k,)} can't be composed
   * and the result degrades to ⊤. The dtype stays precise (float32).
   */
  @Test
  public void testTopkUnknownInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_topk_unknown_input.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Guards the top_k output-shape composer (<a href="https://github.com/wala/ML/issues/609">
   * wala/ML#609</a>): {@code values, indices = tf.math.top_k(x, k=2)} on a {@code (5,)} input
   * yields {@code values} of shape {@code (2,)} float32, composed as {@code input.shape[:-1] +
   * (k,)} by {@link com.ibm.wala.cast.python.ml.client.TopK} rather than left at ⊤. Destructuring
   * (not the wala/ML#480 attribute-access path) gives the precise per-element type.
   */
  @Test
  public void testTopkShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_topk_shape.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/603">wala/ML#603</a>: slicing a
   * NamedTuple result ({@code tf.math.top_k}) walks an object catalog whose keys include the string
   * field aliases {@code values}/{@code indices} alongside the integer element indices. Those
   * non-integer keys must be filtered rather than crashing {@code getFieldIndex}; the slice then
   * recovers the element dtypes ({@code float32} values, {@code int32} indices). No {@code
   * read_data} is involved, so this is a case wala/ML#380 would not fix.
   *
   * <p>The composed {@code (k,) = (2,)} shape now appears for both elements (the wala/ML#609
   * composer), so the asserted set is a union of the precise {@code (2,)} shapes and the residual ⊤
   * ones. The ⊤ components come from the wala/ML#480 attribute/slice path, which doesn't carry the
   * composed per-element shape through; once wala/ML#480 lands they drop, narrowing this to {@code
   * Set.of(TENSOR_2_FLOAT32, TENSOR_2_INT32)}.
   */
  @Test
  public void testTopkSliceCatalog()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_topk_slice_catalog.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TENSOR_UNKNOWN_SHAPE_FLOAT32,
                TENSOR_2_FLOAT32,
                TENSOR_INT32_UNKNOWN_SHAPE,
                TENSOR_2_INT32)));
  }

  /**
   * Guards {@code tf.fill}'s {@code .shape}-argument recovery (<a
   * href="https://github.com/wala/ML/issues/610">wala/ML#610</a>): {@code tf.fill(x.shape, 5.0)}
   * where {@code x} is {@code (2, 2)} yields {@code (2, 2) float32}. The {@code dims} argument is a
   * {@code .shape} property read with an empty points-to set, so resolution falls to {@link
   * com.ibm.wala.cast.python.ml.client.Fill#getDefaultShapes}, which recovers the source tensor's
   * shape rather than dropping to ⊤.
   */
  @Test
  public void testFillShapeArg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_fill_shape_arg.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/606">wala/ML#606</a>: when
   * {@code tf.fill}'s {@code dims} argument is unresolvable (here from {@code json.loads}), {@link
   * com.ibm.wala.cast.python.ml.client.Fill#getDefaultShapes} must return ⊤ rather than throwing
   * {@code UnsupportedOperationException}, which previously aborted the whole analysis. {@code
   * Fill} extends {@code Constant}, so the base allocator floor (wala/ML#604) doesn't cover it. The
   * result is ⊤-shape {@code int32} (the fill value's dtype).
   */
  @Test
  public void testFillUnresolvableDims()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_fill_unresolvable_dims.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * Guards allocator-shape recovery from a {@code .shape} argument (<a
   * href="https://github.com/wala/ML/issues/604">wala/ML#604</a>): {@code tf.ones(x.shape)} where
   * {@code x} is {@code (2, 2)} yields {@code (2, 2) float32}. The shape argument is a {@code
   * .shape} property read with an empty points-to set, so resolution falls to {@link
   * com.ibm.wala.cast.python.ml.client.TensorTypeAllocator#getDefaultShapes}, which recovers the
   * source tensor's shape via {@code getShapeFromShapeAttributeArgument} rather than dropping to ⊤.
   */
  @Test
  public void testOnesTensorShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ones_tensor_shape.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Guards the {@code tf.eye} unresolvable-{@code num_rows} floor (<a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>): when {@code num_rows} is
   * unresolvable (here from {@code json.loads}), the result is still a rank-2 square matrix, so it
   * floors to {@code (Dynamic, Dynamic)} float32 rather than throwing "num_rows parameter is
   * required" (which previously aborted the whole analysis).
   */
  @Test
  public void testEyeUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_eye_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_DYNAMIC_DYNAMIC_FLOAT32)));
  }

  /**
   * Complements {@link #testEyeUnresolvableBatchShape()}: when {@code batch_shape} is a list
   * literal whose length (and hence the output rank) is statically known but whose element is
   * unresolvable (here from {@code json.loads}), precision is preserved rather than floored to ⊤.
   * The single leading batch dimension is dynamic and the {@code (num_rows, num_columns)} suffix
   * stays exact, so {@code tf.eye(3, batch_shape=[<unknown>])} types to {@code (Dynamic, 3, 3)}
   * float32. See <a href="https://github.com/wala/ML/issues/611">wala/ML#611</a>.
   */
  @Test
  public void testEyeDynamicBatch()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t =
        new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(3), new NumericDim(3)));
    test("tf2_test_eye_dynamic_batch.py", "consume", 1, 1, Map.of(2, Set.of(t)));
  }

  /**
   * Guards the {@code tf.eye} unresolvable-{@code batch_shape} floor (<a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>): when {@code batch_shape} is
   * present but unresolvable (here from {@code json.loads}), the number of leading batch dimensions
   * is unknown, so the overall rank can't be known and the result floors to ⊤ rather than throwing
   * "Batch shape argument for tf.eye() should be a list of dimensions." (which previously aborted
   * the whole analysis). The dtype stays float32.
   *
   * <p>TODO: the {@code batch_shape} value here is content-dependent (it comes from {@code
   * json.loads}), so recovering it is the user-annotation problem tracked by <a
   * href="https://github.com/wala/ML/issues/370">wala/ML#370</a>, the same recovery path the
   * allocator shape floor points at. Orthogonally, a structurally-inferable tensor {@code
   * batch_shape} (e.g. {@code tf.shape(x)}, whose rank, and often values, are statically known) can
   * be recovered without an annotation; that is tracked by <a
   * href="https://github.com/wala/ML/issues/619">wala/ML#619</a>. The already-recoverable
   * known-rank case is guarded by {@link #testEyeDynamicBatch()}.
   */
  @Test
  public void testEyeUnresolvableBatchShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_eye_unresolvable_batch_shape.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Guards the {@code tf.random.gamma} unresolvable-{@code shape} floor (<a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>): when the {@code shape} argument
   * is unresolvable (here from {@code json.loads}) the output rank can't be known, so the result
   * floors to ⊤ rather than throwing (which previously aborted the whole analysis). The dtype stays
   * float32.
   */
  @Test
  public void testGammaUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gamma_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Guards the {@code tf.random.poisson} unresolvable-{@code shape} floor (<a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>): same as {@link
   * #testGammaUnresolvable()}, the output rank rides on the unresolvable {@code shape}, so the
   * result floors to ⊤ rather than throwing.
   */
  @Test
  public void testPoissonUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_poisson_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Regression for the {@code getIntValueFromInstanceKey} non-numeric-constant degradation (<a
   * href="https://github.com/wala/ML/issues/590">wala/ML#590</a>): {@code tf.eye(True)} models the
   * Python {@code bool} as a {@code Boolean} constant, which degrades to {@code int(True) == 1} (a
   * {@code (1, 1)} identity) rather than throwing a {@code ClassCastException} on the {@code
   * Number} cast.
   */
  @Test
  public void testEyeBoolDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye_bool_dim.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_1_1_FLOAT32)));
  }

  /**
   * Companion to {@link #testEyeBoolDim} covering the {@code int(False) == 0} branch of {@code
   * getIntValueFromInstanceKey} (<a href="https://github.com/wala/ML/issues/590">wala/ML#590</a>):
   * {@code tf.eye(False)} degrades the {@code Boolean} to {@code 0}, yielding a {@code (0, 0)}
   * identity.
   */
  @Test
  public void testEyeBoolDimFalse()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye_bool_dim_false.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_0_0_FLOAT32)));
  }

  /**
   * Captured-gap regression for the {@code RaggedFromNested} shape floor (<a
   * href="https://github.com/wala/ML/issues/612">wala/ML#612</a>): {@code
   * tf.RaggedTensor.from_nested_row_lengths} with an opaque (unresolvable) {@code
   * nested_row_lengths} floors the shape to ⊤ (unknown) rather than aborting the whole analysis
   * with "Could not calculate shapes". The dtype still resolves to {@code int32} from the {@code
   * flat_values}.
   */
  @Test
  public void testRaggedFromNestedRowLengthsUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_lengths_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * Captured-gap regression for the {@code RaggedFromNestedValueRowIds} shape and dtype floors (<a
   * href="https://github.com/wala/ML/issues/612">wala/ML#612</a>): {@code
   * tf.RaggedTensor.from_nested_value_rowids} with opaque (unresolvable) {@code flat_values} and
   * {@code nested_value_rowids} floors both the shape and the dtype to ⊤ rather than aborting with
   * "Could not calculate shapes" / "Could not determine dtypes".
   */
  @Test
  public void testRaggedFromNestedValueRowIdsUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_value_rowids_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Captured-gap regression for the {@code RaggedConstant} shape and dtype floors (<a
   * href="https://github.com/wala/ML/issues/612">wala/ML#612</a>): {@code tf.ragged.constant} whose
   * {@code pylist} comes from an unmodeled {@code json.loads} &mdash; so its points-to set is empty
   * even though the values are inline &mdash; floors both the shape and the dtype to ⊤ rather than
   * aborting with "Empty points-to set".
   *
   * <p>TODO: The runtime tensor is {@code (2, None)} {@code int32} (asserted in the fixture); the
   * static result floors both axes to ⊤ because {@code json.loads} is unmodeled. This is a modeling
   * gap, not a content-dependent (opaque) value, tracked by <a
   * href="https://github.com/wala/ML/issues/536">wala/ML#536</a> (model {@code json.loads} for
   * compile-time-constant string inputs). ⊤ is the correct floor until then; it is not on the
   * input-signature eval path.
   */
  @Test
  public void testRaggedConstantUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Captured-gap regression for the {@code RaggedConstant} shape and dtype floors (<a
   * href="https://github.com/wala/ML/issues/612">wala/ML#612</a>) along the structural-walk path: a
   * {@code pylist} whose outer list is resolvable but whose first element is an {@code np.ndarray}
   * (neither a {@code list} nor a {@code tuple}) floors both the shape and the dtype to ⊤ rather
   * than aborting the whole analysis with "Expected a list or tuple". Complements {@link
   * #testRaggedConstantUnresolvable()}, which exercises the empty-points-to-set floor.
   *
   * <p>TODO: The runtime tensor is {@code (2, None)} {@code int32} (asserted in the fixture). The
   * dtype floor is inherent until numpy dtype-promotion is modeled (<a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>): an {@code np.ndarray} element's
   * dtype is soundly ⊤ (numpy promotes {@code int} to {@code int64}, not {@code int32}), so the
   * union floors to ⊤. The shape floor reflects the unmodeled ragged rank over a tensor row;
   * delegating the element to its producer generator would recover the shape (<a
   * href="https://github.com/wala/ML/issues/652">wala/ML#652</a>).
   */
  @Test
  public void testRaggedConstantUnresolvableElement()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant_unresolvable_element.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Captured-gap regression for the {@code RaggedConstant} shape floor (<a
   * href="https://github.com/wala/ML/issues/612">wala/ML#612</a>) along the depth-walk path, and a
   * precision guard for the dtype. A {@code pylist} whose first row is a resolvable scalar list but
   * whose second row is an {@code np.ndarray} trips the structural floor in {@code
   * getMaximumDepthOfScalars} (a different site than {@link
   * #testRaggedConstantUnresolvableElement()}, which trips {@code containsScalars}), flooring the
   * shape to ⊤. The dtype is still resolved to {@code int32}, because the leading scalar row lets
   * {@code getDefaultDTypes} confirm scalars before the {@code np.ndarray} element &mdash; so the
   * floor is not the all-⊤ result of {@link #testRaggedConstantUnresolvableElement()}, where the
   * {@code np.ndarray} element precedes any confirmable scalar.
   *
   * <p>TODO: The runtime shape is {@code (2, None)} (asserted in the fixture); the static shape
   * floors to ⊤ over the {@code np.ndarray} row (the unmodeled ragged rank over a tensor element).
   * Delegating the element to its producer generator would recover the shape (<a
   * href="https://github.com/wala/ML/issues/652">wala/ML#652</a>).
   */
  @Test
  public void testRaggedConstantUnresolvableDepth()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant_unresolvable_depth.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * Covers {@code tf.eye} with a {@code batch_shape}, which prepends the batch dimensions to the
   * identity shape (<a href="https://github.com/wala/ML/issues/591">wala/ML#591</a>): a {@code (3,
   * 3)} identity with {@code batch_shape=[2]} is {@code (2, 3, 3)}. Exercises the fresh-list
   * construction that replaced the shared-list mutation.
   */
  @Test
  public void testEyeBatchShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye_batch_shape.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_3_FLOAT32)));
  }

  /**
   * Covers the all-numeric {@link TensorType#of} factories (<a
   * href="https://github.com/wala/ML/issues/594">wala/ML#594</a>): the {@code DType} and {@code
   * String} cell-type overloads map {@code int} dimensions to {@link NumericDim}s, equivalent to
   * the explicit-{@code List} construction, and compose with {@link TensorType#asSparse()}.
   */
  @Test
  public void testTensorTypeNumericFactory() {
    assertEquals(
        new TensorType(FLOAT32, asList(new NumericDim(2), new NumericDim(3))),
        TensorType.of(FLOAT32, 2, 3));
    assertEquals(new TensorType(FLOAT_32, asList(new NumericDim(3))), TensorType.of(FLOAT_32, 3));
    assertTrue(TensorType.of(FLOAT32, 2, 2).asSparse().isSparse());
  }

  /**
   * Guards constant-step subscript-slice shape propagation on ndarrays (wala/ML#405): {@code
   * x_train[:5]} on a {@code (60000, 28, 28) uint8} ndarray yields a {@code (5, 28, 28) uint8}
   * tensor. Implemented via {@link SliceBuiltinOperation}; the receiver-shape leak that previously
   * forced this suppression is closed by the set-shape edge-transfer pin on subscript-result
   * variables in {@link PythonTensorAnalysisEngine}.
   */
  @Test
  public void testSubscriptSlicePreservesShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_subscript_slice_preserves_shape.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_5_28_28_UINT8)));
  }

  /**
   * Regression guard for wala/ML#358.
   *
   * <p>Derived from {@link #testModelCall()} (see {@code tf2_test_model_call.py}) by adding a
   * {@code consume(x)} call inside {@code SequentialModel.__call__} immediately after {@code x =
   * self.dense_2(x)}. At that point {@code x} has shape {@code (20, 10)} and dtype {@code float32}:
   * the chain traces {@code (20, 28, 28)} input → {@code Flatten} → {@code (20, 784)} → 100× {@code
   * Dense(64)} → {@code Dropout} → {@code Dense(10)} → {@code (20, 10)}. {@link
   * com.ibm.wala.cast.python.ml.client.DenseCall#getDefaultShapes} recovers the input shape via an
   * SSA-chain fallback when the PTS walk's allocating-node dispatch can't identify the upstream
   * layer call (Flatten, Dropout, or another Dense).
   */
  @Test
  public void testModelCallConsume()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call_consume.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_20_10_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/599">wala/ML#599</a>.
   *
   * <p>Derived from {@link #testModelCall()} (see {@code tf2_test_model_call.py}) by adding a
   * {@code consume(x)} call <em>inside</em> the {@code for layer in self.my_layers} loop,
   * immediately after {@code x = layer(x)}. At that point {@code x} is the loop's {@code Dense(64)}
   * output, shape {@code (20, 64)} and dtype {@code float32}.
   *
   * <p>This pins the loop-iterated layer call's output, which is the gap wala/ML#599 closes:
   * because {@code range(n)} now returns an iterable (non-empty) list, the {@code self.my_layers}
   * comprehension populates the list, the {@code self.my_layers[idx]} subscript read resolves to
   * its {@code Dense(64)} elements, and the loop call's output narrows to {@code (20, 64)} rather
   * than carrying the upstream {@code Flatten} shape {@code (20, 784)} unchanged. {@link
   * com.ibm.wala.cast.python.ml.client.DenseCall} breaks the resulting input-shape self-recursion
   * (the loop's collapsed 1-CFA node feeds its own output back in) via a per-thread cycle guard.
   */
  @Test
  public void testModelCallLoopConsume()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_model_call_loop_consume.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_20_64_FLOAT32)));
  }

  /**
   * Regression guard for the "layer-output flows into script-level consumer" pattern: a downstream
   * function whose tensor parameter comes from a layer call's output. Companion to {@link
   * #testModelCallConsume()}, which calls {@code consume(x)} <em>inside</em> {@code
   * SequentialModel.__call__}; this fixture calls {@code consume(pred)} at script-level after the
   * layer call — the same surface shape, but a different caller. The existing {@code
   * DenseCall.getDefaultShapes} SSA-chain fallback recovers the result type when the direct PTS
   * walk doesn't carry the synthetic {@code <new>} alloc through.
   */
  @Test
  public void testLayerOutputParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_layer_output_param.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_1_10_FLOAT32)));
  }

  /**
   * Variant of {@link #testLayerOutputParam()} that interposes a user-defined {@code
   * tf.keras.Model} subclass between the {@code Dense} layers and the script-level consumer —
   * exercises the same fallback through one extra level of call indirection.
   */
  @Test
  public void testLayerOutputParamViaModel()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_layer_output_param_via_model.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_10_FLOAT32)));
  }

  private void test(
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
    LOGGER.info("Tensor analysis: " + analysis);

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
                                + pointerKeyToTensorVariable.get(pk))
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

  /**
   * Regression guard for wala/ML#646: a SparseTensor flows through a dict subscript ({@code
   * features["t"]}) and keeps its type. {@code tf.sparse.SparseTensor} allocates directly in {@code
   * do()} (the former {@code read_data} call was inlined), so the result carries a live points-to
   * set that survives the dict {@code putfield}/{@code getfield}; the earlier empty PTS dropped it.
   */
  @Test
  public void testSparseTensorThroughDict()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dict_subscript.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 2, 2).asSparse())));
  }
}
