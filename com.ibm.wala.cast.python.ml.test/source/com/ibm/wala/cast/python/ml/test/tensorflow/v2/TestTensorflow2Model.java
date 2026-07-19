package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_STRING;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_1_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_28_28_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_32_32_3_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_102_13_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_102_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10_2_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_16_28_28_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_0_0_9_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_10_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_2_27_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2246_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2246_OBJECT;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_25000_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_25000_OBJECT;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_4_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_6_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_30_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_32_28_28_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_5_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_STRING;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_404_13_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_404_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4096_32_32_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4096_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_8_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_50000_1_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_50000_32_32_3_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_28_28_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_28_28_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_784_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_6_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_6_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_7_5_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_8982_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_8982_OBJECT;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_DYNAMIC_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_INT32_UNKNOWN_SHAPE;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.COMPLEX64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.BroadcastTo;
import com.ibm.wala.cast.python.ml.client.Linspace;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConcreteTypeKey;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.intset.OrdinalSet;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/** Test TF2 APIs. */
public class TestTensorflow2Model extends AbstractTensorTest {

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

  /**
   * {@code add(a, b)} is called only as {@code add(list, list)} where {@code list = [tf.ones([1,
   * 2]), tf.ones([2, 2])]}, so {@code a} and {@code b} are Python lists. {@code a + b} is therefore
   * list concatenation, not a tensor add, so {@code add} has no tensor variables. Before
   * wala/ML#750 the {@code list} operands' allocations were counted as tensor evidence and {@code a
   * + b} was falsely typed as a tensor (the local count was 1); the concatenation is now correctly
   * not a tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testTensorList2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tensor_list2.py", "add", 0, 0);
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
   *     <p>The local count excludes the {@code enumerate(targets)} iterator object, pinned
   *     non-tensor since wala/ML#732; the element and its TensorFlow uses stay typed.
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
        6,
        Map.of(
            3,
            Set.of(
                new TensorType(INT_32, asList(new NumericDim(2), DynamicDim.INSTANCE)),
                new TensorType(INT_32, asList(new SymbolicDim("?"), DynamicDim.INSTANCE)))));
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
    test("tf2_test_top_p_logits.py", "top_p_logits", 1, 12, Map.of(2, Set.of(TENSOR_1_5_FLOAT32)));
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
        10,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32), 3, Set.of(TENSOR_2_2_INT32)));
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
   * A negative k-CFA depth is invalid (the depth is the call-string length for the targeted context
   * selector) and is rejected at construction (wala/ML#379).
   */
  @Test(expected = IllegalArgumentException.class)
  public void testNegativeTargetedCfaDepthRejected() {
    new PythonTensorAnalysisEngine(
        emptyList(), PythonTensorAnalysisEngine.TENSORFLOW, /* targetedCfaDepth= */ -1);
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

  /**
   * Regression test for <a href="https://github.com/wala/ML/issues/750">wala/ML#750</a>: a Python
   * list concatenation accumulated in a loop ({@code result_array += [on, off]}) is not a tensor.
   * The pattern is vendored from {@code MusicTransformer-tensorflow2.0}'s {@code
   * midi_processor/processor.py} {@code _divide_note}, where {@code on}/{@code off} are plain
   * {@code SplitNote} objects and {@code result_array} is a list. The concatenation's {@code +}
   * binop lands in {@link
   * com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine#getDataflowSources}'s {@code
   * SSABinaryOpInstruction} branch, so it is gated by operand tensor evidence. The loop-carried
   * {@code result_array} is a phi over the initial {@code []} allocation and the previous
   * concatenation, so its {@code list} allocation is reached only through its points-to set, not
   * its own def. Before the fix, the points-to-set tensor-evidence check counted that {@code list}
   * allocation as evidence, so the binop dispatched to {@code ElementWiseOperation} and typed the
   * result as a tensor; that type then fed back through the loop into {@code result_array}, a
   * self-reinforcing false tensor with a {@code TENSORFLOW} origin. {@code _divide_note} must have
   * no tensor variables.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testListConcatNotTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf750_list_concat.py", "_divide_note", 0, 0);
  }

  /**
   * Companion to {@link #testListConcatNotTensor()} for the {@code tuple} arm of the wala/ML#750
   * fix. {@code _divide_note_tuple} accumulates a loop-carried {@code tuple} with {@code
   * result_tuple += (on, off)}. Like the list case, the {@code +} binop is gated by operand tensor
   * evidence, and the loop-carried {@code tuple}'s allocation is reached only through its points-to
   * set. A {@code tuple} concatenation is not a tensor op, so the accumulator's {@code tuple}
   * allocation must not count as tensor evidence and {@code _divide_note_tuple} must have no tensor
   * variables.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testTupleConcatNotTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf750_list_concat.py", "_divide_note_tuple", 0, 0);
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
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>: a tensor
   * passed interprocedurally to a callee types the callee's parameter. {@code Model.get_loss}'s
   * {@code real} and {@code pred} receive {@code tf.constant} tensors via {@code train_step}, so
   * both type to {@code (3,)} float32 rather than being missed. With the widened operand walks
   * (wala/ML#739), all three body producers count as tensor locals with their runtime-true shapes:
   * {@code pred - real} and {@code tf.square} at {@code (3,)} and the {@code tf.reduce_mean} result
   * at scalar rank 0.
   */
  @Test
  public void testInterprocTensorParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = TensorType.of(FLOAT_32, 3);

    test(
        "tf2_test_interproc_tensor_param.py",
        "Model.get_loss",
        2,
        5,
        Map.of(3, Set.of(t), 4, Set.of(t)));
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

  // wala/ML#449 Tier 3: reductions. Each collapses dims along `axis` (default `None` = all dims)
  // and preserves dtype from input (except `reduce_all` which is always BOOL).

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
   * Test two call sites of the same method with equal total arity but different positional/keyword
   * splits. The trampoline cache must not collide them; see <a
   * href="https://github.com/wala/ML/issues/740">wala/ML#740</a>. The parameter {@code x} receives
   * a tensor positionally at one call site and by keyword at the other, so its type must be the
   * union of both.
   */
  @Test
  public void testTrampolineSplit() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_trampoline_split.py",
        "C.f",
        1,
        1,
        Map.of(3, Set.of(TENSOR_1_2_FLOAT32, TENSOR_1_3_FLOAT32)));
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
   * Distilled regression guard for the scalar-expression broadcast (wala/ML#718): a tensor scaled
   * by a statically opaque scalar expression ({@code 1.0 / math.sqrt(4.0)}) keeps its shape, since
   * the expression's scalarness is structural even though its value never resolves. The analysis
   * previously erased the result to ⊥. Mirrors NLPGNN's attention-logit scaling.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testScalarExpressionScale()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_scalar_scale.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4))));
  }

  /**
   * Distilled guard for the dimension-provenance split (wala/ML#721): a configuration-sourced size
   * — here an environment read the analysis cannot fold — types as {@link UnresolvedDim}, a fixed
   * runtime size of unknown value, not {@link DynamicDim}, which is reserved for axes the runtime
   * {@code TensorShape} reports as {@code None}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testUnresolvedDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_unresolved_dim.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(100))))));
  }

  /**
   * Pins the fold taint of the dimension-provenance split (wala/ML#721): {@code np.prod} over a
   * shape list whose leading axis is {@code None} stays {@link DynamicDim} — arithmetic over a
   * {@code None} axis is itself {@code None} at run time — rather than degrading to {@link
   * UnresolvedDim}. The runtime guards the {@code None} away before folding (the BERT {@code
   * get_shape_list} idiom), but the static walk sees the unguarded shape. The {@code (24)} member
   * is the guarded arm — the runtime-true fold over the patched {@code [1, 4, 6]} — flowing
   * alongside per the guard-φ.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testProdOverDynamic()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_prod_over_dynamic.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE)),
                TensorType.of(FLOAT_32, 24))));
  }

  /**
   * Distilled regression guard for {@code tf.matmul}'s batched form (wala/ML#718): the leading
   * (batch) dimensions carry through and the trailing two dimensions compose as the matrix product,
   * so the rank is preserved. The analysis previously collapsed every product to rank two.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBatchedMatMul()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_batched_matmul.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 4, 5))));
  }

  /**
   * NLPGNN's {@code WDEmbedding} in miniature (wala/ML#711, wala/ML#717): the output reshape's
   * target slices the rank-conditional {@code tf.expand_dims} guard's φ, so the composition
   * cross-products the two source shapes per position: four members, including the runtime {@code
   * (2, 2, 8)}, with the trailing dimension static in every member. The mixed-pairing members are
   * the cross-product's sound over-approximation; pairing the evaluation per φ member would drop
   * them. The embedding table's {@code float32} survives through both the {@code gather} arm and
   * the {@code one_hot}/{@code matmul} arm.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEmbeddingOutput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_embedding_output.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 16),
                TensorType.of(FLOAT_32, 2, 8),
                TensorType.of(FLOAT_32, 2, 2, 16),
                TensorType.of(FLOAT_32, 2, 2, 8))));
  }

  /**
   * Opaque-size variant of {@link #testEmbeddingOutput()} (wala/ML#717), mirroring the vendored
   * NLPGNN embedding, whose table size comes from a checkpoint config the analysis cannot read. The
   * output reshape's trailing element {@code input_shape[-1] * self.embedding_size} then has an
   * unresolvable factor, but arithmetic over a shape-vector subscript is one scalar dimension, so
   * the value degrades to dynamic while the rank and the leading dimensions survive, per guard-φ
   * member.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testEmbeddingDynamicSize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_embedding_dynamic_size.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, asList(new NumericDim(2), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32,
                    asList(new NumericDim(2), new NumericDim(2), UnresolvedDim.INSTANCE)))));
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
}
