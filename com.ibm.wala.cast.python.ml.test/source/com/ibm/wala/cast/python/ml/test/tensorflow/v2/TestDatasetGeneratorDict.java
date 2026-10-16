package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a dataset built by {@code from_generator} with a dict of {@code output_types} types
 * each key of its element, kept through {@code padded_batch}, where a function reads the key and
 * expands it. Before, a string key into such an element resolved nothing, and the value read as a
 * tensor of unknown dtype.
 */
public class TestDatasetGeneratorDict extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_generator_dict.py";

  /** A dataset of file names whose maps load them into a dict of arrays. */
  private static final String LOAD_MAP_FILE = "tf2_test_dataset_generator_load_map.py";

  /**
   * A padded batch of 1-D {@code float32} values, expanded to rank 3: a full batch of two, or a
   * last, partial one.
   */
  private static final Set<TensorType> EXPANDED_BATCH =
      Set.of(
          new TensorType(
              FLOAT_32, asList(new NumericDim(2), DynamicDim.INSTANCE, new NumericDim(1))),
          new TensorType(
              FLOAT_32, asList(new SymbolicDim("?"), DynamicDim.INSTANCE, new NumericDim(1))));

  /** A batch of rows of three {@code float32} values. */
  private static final TensorType BATCH_3 = TensorType.of(FLOAT_32, 2, 3);

  /** The dtypes alone, with the shapes from {@code padded_shapes}. */
  @Test
  public void testOutputTypesDict() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "types_only_sink", 1, 2, Map.of(2, EXPANDED_BATCH));
  }

  /** The dtypes with a dict of {@code output_shapes} too. */
  @Test
  public void testOutputTypesAndShapesDict()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "shaped_sink", 1, 2, Map.of(2, EXPANDED_BATCH));
  }

  /**
   * A dataset class's {@code from_generator}, {@code filter}, {@code padded_batch} and {@code
   * prefetch}, distributed and stepped by a trainer through a {@code tf.function} that runs the
   * step per replica.
   */
  @Test
  public void testTrainerStep() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_dataset_generator_trainer.py", "trainer_sink", 1, 2, Map.of(2, EXPANDED_BATCH));
  }

  /** A dataset an object holds in a field, enumerated in a method: the element is typed. */
  @Test
  public void testEnumerateFieldDataset()
      throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_enumerate_field_dataset.py", "field_sink", 1, 2, Map.of(2, Set.of(BATCH_3)));
  }

  /** Control: the same dataset enumerated from a local variable. */
  @Test
  public void testEnumerateLocalDataset()
      throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_enumerate_field_dataset.py", "local_sink", 1, 2, Map.of(2, Set.of(BATCH_3)));
  }

  /**
   * A key whose {@code padded_shapes} entry cannot be read: its shape is unknown, not the upstream
   * component's, which the padding may change.
   */
  @Test
  public void testUnreadPaddedShape() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "unread_padding_sink", 1, 2, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * A trainer whose loader attribute is written three times, as a common base trainer writes it:
   * {@code None} at construction, the dataset a setter is passed, and the distributed dataset made
   * from the attribute's own value. The element of the enumerated loader is any of the datasets
   * written, so it is their join.
   */
  @Test
  public void testBaseTrainerLoader() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_dataset_generator_trainer.py",
        "base_trainer_sink",
        1,
        2,
        Map.of(2, EXPANDED_BATCH));
  }

  /** The dtypes and shapes from a dict of specs, the {@code output_signature} form. */
  @Test
  public void testOutputSignatureDict()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "signature_sink", 1, 2, Map.of(2, EXPANDED_BATCH));
  }

  /**
   * An attribute written with a dataset and with a distributed view of another object's attribute:
   * the second write brings the other object's datasets, so it is no rewrite of this attribute's
   * own value. Its part resolves no element here, so the element is not typed, rather than typed as
   * the first write's dataset alone.
   */
  @Test
  public void testOtherObjectsAttribute()
      throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_dataset_generator_trainer.py", "copier_sink", 0, 0, Map.of());
  }

  /** A padded batch without {@code padded_shapes}: each key pads to the longest in its batch. */
  @Test
  public void testPaddedBatchWithoutPaddedShapes()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "unpadded_sink",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, asList(new NumericDim(2), DynamicDim.INSTANCE)),
                new TensorType(FLOAT_32, asList(new SymbolicDim("?"), DynamicDim.INSTANCE)))));
  }

  /** A declared shape the reader cannot read: the key's shape is unknown, its dtype declared. */
  @Test
  public void testUnreadOutputShape() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "unread_shape_sink", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * An attribute two calls write datasets of dicts to, enumerated and read by key: the element is
   * either dataset's.
   */
  @Test
  public void testJoinedDatasetsByKey()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_enumerate_join.py",
        "dict_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(INT_32, 2, 3))));
  }

  /**
   * An attribute two calls write datasets of tuples to, enumerated and read by index: the element
   * is either dataset's.
   */
  @Test
  public void testJoinedDatasetsByIndex()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_enumerate_join.py",
        "tuple_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(INT_32, 2, 3))));
  }

  /**
   * The same joined tuple element expanded: the expansion reads the element's shapes and dtypes by
   * index from the join.
   */
  @Test
  public void testJoinedDatasetsExpanded()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_enumerate_join.py",
        "expanded_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3, 1), TensorType.of(INT_32, 2, 3, 1))));
  }

  /** A dataset the class body assigns, enumerated through an instance and read by key. */
  @Test
  public void testClassAttributeLoader()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_enumerate_class_attribute.py",
        "class_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3))));
  }

  /**
   * A dataset the class body assigns beside one an instance write stores: the element is either,
   * not the instance write's alone.
   */
  @Test
  public void testClassAndInstanceAttributeLoader()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_enumerate_class_attribute.py",
        "mixed_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3), TensorType.of(INT_32, 2, 3))));
  }

  /**
   * A class of its own whose body assigns the same attribute name a {@code bool} dataset: its
   * element is that dataset's, and it reaches no other class's read.
   */
  @Test
  public void testUnrelatedClassAttributeLoader()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_enumerate_class_attribute.py",
        "unrelated_sink",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(BOOL, 2, 3))));
  }

  /**
   * A dataset whose {@code from_generator} element holds file names, whose {@code map} loads them
   * into a new dict of arrays and a second {@code map} rebuilds that dict, padded and batched, and
   * read by key in a trainer: the key is typed from the mapped dict, padded by its own {@code
   * padded_shapes} entry. Before, the key read was the union of every entry of {@code
   * padded_shapes}, of unknown dtype.
   */
  @Test
  public void testLoadingMapKey() throws ClassHierarchyException, CancelException, IOException {
    test(LOAD_MAP_FILE, "load_sink", 1, 2, Map.of(2, EXPANDED_BATCH));
  }

  /** Another key of the same batch, padded to two axes. */
  @Test
  public void testLoadingMapOtherKey()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        LOAD_MAP_FILE,
        "mels_sink",
        1,
        2,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    FLOAT_32,
                    asList(
                        new NumericDim(2),
                        DynamicDim.INSTANCE,
                        new NumericDim(80),
                        new NumericDim(1))),
                new TensorType(
                    FLOAT_32,
                    asList(
                        new SymbolicDim("?"),
                        DynamicDim.INSTANCE,
                        new NumericDim(80),
                        new NumericDim(1))))));
  }

  /** The same key, read from a batch the dataset is enumerated into directly. */
  @Test
  public void testLoadingMapKeyEnumerated()
      throws ClassHierarchyException, CancelException, IOException {
    test(LOAD_MAP_FILE, "local_sink", 1, 2, Map.of(2, EXPANDED_BATCH));
  }

  /**
   * A key of a dict the map function builds by calling {@code dict}, whose keys the analysis does
   * not read: the key's shape is its own {@code padded_shapes} entry, not the union of every entry,
   * and its dtype, which the built dict would give, is unknown.
   */
  @Test
  public void testLoadingMapBuiltDict()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        LOAD_MAP_FILE,
        "built_sink",
        1,
        2,
        Map.of(
            2,
            Set.of(
                new TensorType(
                    UNKNOWN, asList(new NumericDim(2), DynamicDim.INSTANCE, new NumericDim(1))),
                new TensorType(
                    UNKNOWN,
                    asList(new SymbolicDim("?"), DynamicDim.INSTANCE, new NumericDim(1))))));
  }

  /**
   * A key of a dict the map function writes into its input and returns, which neither a dict the
   * map function returns nor a {@code padded_shapes} entry names: the key's type is unknown, not
   * the union of every key's.
   */
  @Test
  public void testUnresolvedKey() throws ClassHierarchyException, CancelException, IOException {
    test(LOAD_MAP_FILE, "rewritten_sink", 1, 1, Map.of(2, Set.of(new TensorType(UNKNOWN, null))));
  }

  /**
   * A key of the dict a map function returns from two sites, one storing a loaded array and one a
   * value it computes: either site's value.
   */
  @Test
  public void testMapDictReturnSites()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_dataset_map_dict_sites.py",
        "sites_sink",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_32, null), new TensorType(INT_32, null))));
  }

  /**
   * A key of the dict a map function returns, holding a tensor it allocates, read from a batch
   * without {@code padded_shapes}: the tensor's own shape under the batch axis.
   */
  @Test
  public void testMapDictAllocatedKey()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        LOAD_MAP_FILE,
        "concrete_sink",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 3),
                new TensorType(FLOAT_32, asList(new SymbolicDim("?"), new NumericDim(3))))));
  }

  /**
   * A key of the same dict holding a value the map function computes, which has no allocation: the
   * computed value's shape under the batch axis.
   */
  @Test
  public void testMapDictComputedKey()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        LOAD_MAP_FILE,
        "computed_sink",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(INT_32, 2), new TensorType(INT_32, asList(new SymbolicDim("?"))))));
  }

  /** A dataset of dicts whose map function passes each key's value on (wala/ML#1088). */
  private static final String DICT_MAP_FILE = "tf2_test_dataset_from_generator_dict_map.py";

  /**
   * The dict a map function is given is no tensor, though its values are (<a
   * href="https://github.com/wala/ML/issues/1088">wala/ML#1088</a>). Before, it was typed as one
   * tensor of the declared dtypes' join.
   */
  @Test
  public void testMapDictParameter() throws ClassHierarchyException, CancelException, IOException {
    test(DICT_MAP_FILE, "load", 0, 4);
  }

  /** A {@code float32} key of the dict, read in the map function. */
  @Test
  public void testMapDictFloatKey() throws ClassHierarchyException, CancelException, IOException {
    test(DICT_MAP_FILE, "consume_mels", 1, 1, Map.of(2, Set.of(new TensorType(FLOAT_32, null))));
  }

  /** A {@code string} key of the dict, read in the map function. */
  @Test
  public void testMapDictStringKey() throws ClassHierarchyException, CancelException, IOException {
    test(DICT_MAP_FILE, "consume_ids", 1, 1, Map.of(2, Set.of(new TensorType(STRING, null))));
  }

  /** The dict a map function is given when an {@code output_signature} dict declares it. */
  @Test
  public void testMapSignatureDictParameter()
      throws ClassHierarchyException, CancelException, IOException {
    test(DICT_MAP_FILE, "load_signed", 0, 2);
  }

  /** A key of the dict an {@code output_signature} dict declares, with its spec's shape. */
  @Test
  public void testMapSignatureDictKey()
      throws ClassHierarchyException, CancelException, IOException {
    test(DICT_MAP_FILE, "consume_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * The batch a map function is given when the batched dataset's elements are dicts: batching a
   * dict batches each key, so the batch is a dict as well.
   */
  @Test
  public void testMapBatchedDictParameter()
      throws ClassHierarchyException, CancelException, IOException {
    test(DICT_MAP_FILE, "load_batched", 0, 2);
  }

  /** A key of a batched dict, read in the map function. */
  @Test
  public void testMapBatchedDictKey() throws ClassHierarchyException, CancelException, IOException {
    test(
        DICT_MAP_FILE,
        "consume_batched_mels",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_32, null))));
  }

  /** As {@link #testMapBatchedDictParameter()}, with {@code padded_batch}. */
  @Test
  public void testMapPaddedDictParameter()
      throws ClassHierarchyException, CancelException, IOException {
    test(DICT_MAP_FILE, "load_padded", 0, 2);
  }

  /** A key of a padded batch of dicts, shaped by its {@code padded_shapes} entry. */
  @Test
  public void testMapPaddedDictKey() throws ClassHierarchyException, CancelException, IOException {
    test(
        DICT_MAP_FILE,
        "consume_padded_mels",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, asList(new SymbolicDim("?"), DynamicDim.INSTANCE)),
                new TensorType(FLOAT_32, asList(new NumericDim(2), DynamicDim.INSTANCE)))));
  }

  /**
   * The batch a trainer's step is given, iterated from a dataset whose dict elements pass through
   * {@code filter}, {@code padded_batch} and {@code prefetch}, is a dict and not a tensor. The
   * transformations read an element that is not a tensor as one whose type did not resolve, so the
   * batch is typed as a tensor of the padded shapes with an unknown dtype.
   *
   * <p>The step has three tensor variables today, one of them the batch parameter: expecting three,
   * the assertion that fails is the tensor-parameter count, one where none is expected. The two
   * expected here are those three less the parameter, derived rather than measured with the
   * parameter untyped.
   *
   * <p>TODO: Not a tensor once https://github.com/wala/ML/issues/1093 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testTrainerStepDictBatch()
      throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_dataset_generator_trainer.py", "BaseTrainer._train_step", 0, 2);
  }

  /**
   * The dict a map function is given is a {@code dict} whose keys hold their declared tensors (<a
   * href="https://github.com/wala/ML/issues/1088">wala/ML#1088</a>), so a client that looks for
   * tensors inside a container parameter finds them. Before, the parameter pointed only to the
   * receiver dataset, so the parameter held neither a tensor nor a container of tensors.
   */
  @Test
  public void testMapDictParameterContainer()
      throws ClassHierarchyException, CancelException, IOException {
    assertEquals(
        Map.of(
            "mels", Set.of(new TensorType(FLOAT_32, null)),
            "utt_ids", Set.of(new TensorType(STRING, null))),
        dictParameterFieldTypes(DICT_MAP_FILE, "load"));
  }

  /**
   * As {@link #testMapDictParameterContainer()}, for a dict an {@code output_signature} declares.
   */
  @Test
  public void testMapSignatureDictParameterContainer()
      throws ClassHierarchyException, CancelException, IOException {
    assertEquals(
        Map.of("x", Set.of(TensorType.of(FLOAT_32, 2))),
        dictParameterFieldTypes(DICT_MAP_FILE, "load_signed"));
  }

  /** As {@link #testMapDictParameterContainer()}, for a batch of dicts. */
  @Test
  public void testMapBatchedDictParameterContainer()
      throws ClassHierarchyException, CancelException, IOException {
    assertEquals(
        Map.of("mels", Set.of(new TensorType(FLOAT_32, null))),
        dictParameterFieldTypes(DICT_MAP_FILE, "load_batched"));
  }

  /** As {@link #testMapDictParameterContainer()}, for a padded batch of dicts. */
  @Test
  public void testMapPaddedDictParameterContainer()
      throws ClassHierarchyException, CancelException, IOException {
    assertEquals(
        Map.of(
            "mels",
            Set.of(
                new TensorType(FLOAT_32, asList(new SymbolicDim("?"), DynamicDim.INSTANCE)),
                new TensorType(FLOAT_32, asList(new NumericDim(2), DynamicDim.INSTANCE)))),
        dictParameterFieldTypes(DICT_MAP_FILE, "load_padded"));
  }

  /**
   * A control: the element of a dataset of tensors is a tensor, so the map function's parameter is
   * typed as one and holds no container of tensors.
   */
  @Test
  public void testMapTensorParameter()
      throws ClassHierarchyException, CancelException, IOException {
    test(DICT_MAP_FILE, "load_scalar", 1, 2, Map.of(2, Set.of(new TensorType(FLOAT_32, null))));
  }

  /** The control of {@link #testMapTensorParameter()}, on the container side. */
  @Test
  public void testMapTensorParameterContainer()
      throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Map.of(), dictParameterFieldTypes(DICT_MAP_FILE, "load_scalar"));
  }

  /**
   * Runs the analysis on a fixture and collects, for each {@code dict} the first parameter of a
   * function points to in any of its nodes, the tensor types on the dict's fields, by field name.
   *
   * @param filename The Python fixture to analyze.
   * @param functionName The function whose parameter is read.
   * @return The types on the fields of the parameter's dicts, by field name.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  private Map<String, Set<TensorType>> dictParameterFieldTypes(String filename, String functionName)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), filename);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    String signature = "script " + filename + "." + functionName + ".do()LRoot;";
    Set<InstanceKey> dicts = HashSetFactory.make();
    boolean reached = false;
    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().equals(signature)) continue;
      reached = true;
      // The first parameter after the function object itself.
      PointerKey param = builder.getPointerKeyForLocal(node, node.getIR().getParameter(1));
      for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(param))
        if (ik instanceof AllocationSiteInNode allocation
            && allocation.concreteType().getReference().equals(PythonTypes.dict)) dicts.add(ik);
    }
    assertTrue("Function must exist in call graph.", reached);

    Map<String, Set<TensorType>> ret = HashMapFactory.make();
    for (Pair<PointerKey, TensorVariable> pair : analysis) {
      if (!(pair.fst instanceof InstanceFieldKey fieldKey)) continue;
      if (!dicts.contains(fieldKey.getInstanceKey())) continue;
      ret.computeIfAbsent(fieldKey.getField().getName().toString(), k -> HashSetFactory.make())
          .addAll(pair.snd.getTypes());
    }
    return ret;
  }
}
