package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.Containers;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.util.collections.Pair;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;
import org.junit.Test;

/**
 * Tests {@link Containers}, the container queries a client makes after the analysis, from outside
 * the analysis's package, on fixtures where each sink receives one container.
 */
public class TestContainers extends TestPythonMLCallGraphShape {

  /** The fixture of tuples, lists and dicts. */
  private static final String FILE = "tf2_test_container_elements.py";

  /** The fixture of dicts, one of them stored under an unresolved key. */
  private static final String DICT_KEYS = "tf2_test_container_elements_dict_keys.py";

  /** Per fixture, its builder. */
  private static final Map<String, PythonSSAPropagationCallGraphBuilder> BUILDERS = new HashMap<>();

  /** Per fixture, its solved analysis. */
  private static final Map<String, TensorTypeAnalysis> ANALYSES = new HashMap<>();

  /**
   * Analyzes a fixture, once.
   *
   * @param file The fixture.
   * @return Its builder.
   */
  private static PythonSSAPropagationCallGraphBuilder builder(String file) {
    return BUILDERS.computeIfAbsent(
        file,
        f -> {
          try {
            PythonTensorAnalysisEngine engine =
                new TestContainers().makeEngine(Collections.<File>emptyList(), f);
            PythonSSAPropagationCallGraphBuilder b = engine.defaultCallGraphBuilder();
            b.makeCallGraph(b.getOptions());
            ANALYSES.put(f, engine.performAnalysis(b));
            return b;
          } catch (Exception e) {
            throw new IllegalStateException(e);
          }
        });
  }

  /**
   * The one container a sink receives.
   *
   * @param file The fixture.
   * @param sink The sink's name.
   * @return The container instance.
   */
  private static InstanceKey container(String file, String sink) {
    PythonSSAPropagationCallGraphBuilder builder = builder(file);
    List<InstanceKey> containers = new ArrayList<>();
    for (CGNode node : builder.getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink)) {
        PointerKey parameter =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
        for (InstanceKey container : builder.getPointerAnalysis().getPointsToSet(parameter))
          if (!containers.contains(container)) containers.add(container);
      }
    assertEquals("Expecting one container at " + sink + ".", 1, containers.size());
    return containers.get(0);
  }

  /**
   * The tensor types the solved analysis gives a key, read as a client reads them.
   *
   * @param file The fixture.
   * @param key The key.
   * @return Its types; empty for a key the analysis gives none.
   */
  private static Set<TensorType> types(String file, PointerKey key) {
    builder(file);
    Set<TensorType> ret = new HashSet<>();
    for (Iterator<Pair<PointerKey, TensorVariable>> it = ANALYSES.get(file).iterator();
        it.hasNext(); ) {
      Pair<PointerKey, TensorVariable> pair = it.next();
      if (pair.fst.equals(key)) ret.addAll(pair.snd.getTypes());
    }
    return ret;
  }

  /**
   * A tuple of two tensors: its length, a key per position whose types a client reads off the
   * analysis, and every element key, with none of them {@code None}.
   */
  @Test
  public void testTuple() {
    PythonSSAPropagationCallGraphBuilder builder = builder(FILE);
    InstanceKey tuple = container(FILE, "sink_tuple");
    assertEquals(OptionalInt.of(2), Containers.getLength(builder, tuple));
    List<PointerKey> positions = Containers.getPositionKeys(builder, tuple);
    assertEquals(2, positions.size());
    assertEquals(Set.of(TensorType.of(FLOAT32, 2, 3)), types(FILE, positions.get(0)));
    assertEquals(Set.of(TensorType.of(INT32, 4)), types(FILE, positions.get(1)));
    assertTrue(Containers.getElementKeys(builder, tuple).containsAll(positions));
    assertTrue(Containers.isComplete(builder, tuple));
    assertFalse(Containers.mayHoldNone(builder, tuple));
  }

  /** An appended list has no length and no position keys, but still has element keys. */
  @Test
  public void testAppended() {
    PythonSSAPropagationCallGraphBuilder builder = builder(FILE);
    InstanceKey list = container(FILE, "sink_appended");
    assertEquals(OptionalInt.empty(), Containers.getLength(builder, list));
    assertTrue(Containers.getPositionKeys(builder, list).isEmpty());
    assertFalse(Containers.getElementKeys(builder, list).isEmpty());
  }

  /** A tuple holding {@code None} may hold it, and one without does not. */
  @Test
  public void testNone() {
    PythonSSAPropagationCallGraphBuilder builder = builder(FILE);
    assertTrue(Containers.mayHoldNone(builder, container(FILE, "sink_with_none")));
    assertFalse(Containers.mayHoldNone(builder, container(FILE, "sink_without_none")));
  }

  /**
   * A dict with a value stored under an unresolved key: its element keys are not all of them, so an
   * element may be {@code None}.
   */
  @Test
  public void testUnresolvedKey() {
    PythonSSAPropagationCallGraphBuilder builder = builder(DICT_KEYS);
    InstanceKey dict = container(DICT_KEYS, "sink_unresolved_key");
    assertFalse(Containers.isComplete(builder, dict));
    assertTrue(Containers.mayHoldNone(builder, dict));
  }
}
