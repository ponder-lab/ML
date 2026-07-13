package com.ibm.wala.cast.python.ml.test;

import static java.util.Collections.emptyList;
import static java.util.Collections.nCopies;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import org.junit.Test;

/**
 * Tests for the origin record on tensor-typed values (wala/ML#724, wala/ML#726).
 *
 * <p>The fixture routes each value of interest through a sink function ({@code consume_np(x)}
 * etc.), per the calling-context convention. A sink's <em>parameter</em> pins the wala/ML#726
 * boundary semantics: it reads exactly its seeded {@link TensorOrigin#PARAMETER}, since the
 * parameter barrier blocks caller-side origin inflow. The routed value's <em>producing</em> origin
 * is therefore observed on the caller-side argument local at each sink call site &mdash; the
 * parameter's immediate dataflow predecessor, and the consumer's view of a def in the calling
 * function's body.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestTensorOrigins extends TestPythonMLCallGraphShape {

  /**
   * The origins observed at a fixture's sink functions.
   *
   * @param parameterOrigins Per sink, the first parameter's origins (value number 2), unioned
   *     across calling contexts; a sink absent from the map received no tensor-typed state at all.
   * @param argumentOrigins Per sink, the first argument's origins at each call site of the sink, in
   *     source order.
   */
  private record SinkOrigins(
      Map<String, Set<TensorOrigin>> parameterOrigins,
      Map<String, List<Set<TensorOrigin>>> argumentOrigins) {}

  /**
   * The three instruction shapes from wala/ML#724 flow to {@code consume_np}: a binary operator
   * ({@code a + b}), an ndarray method ({@code m.reshape(...)}), and an interprocedural return of
   * {@code np.array(...)}. The invoke-produced values are numpy-origin; the operator cases run over
   * their enclosing helpers' parameters, so under wala/ML#726 their results carry the
   * hybridization-frame origin instead (the traced operator is a TensorFlow op whatever the eager
   * feeds). The mixed binary operator ({@code a + t} in {@code mixed}) likewise runs over
   * parameters now that they classify first; the local-level mixed-dispatch pin lives in {@code
   * tf2_test_parameter_origin.py}.
   *
   * @throws ClassHierarchyException If the class hierarchy cannot be built.
   * @throws CancelException If the analysis is canceled.
   * @throws IOException If the test file cannot be read.
   */
  @Test
  public void testNumpyOrigin() throws ClassHierarchyException, CancelException, IOException {
    SinkOrigins sinkOrigins =
        getSinkOrigins(
            "tf2_test_numpy_origin.py",
            "consume_np",
            "consume_np_opaque",
            "consume_np_loader",
            "consume_tf");

    // Every sink parameter reads exactly its seed: the barrier blocks the callers' origins
    // (wala/ML#726), including the six loaders' numpy inflow.
    for (String sink :
        List.of("consume_np", "consume_np_opaque", "consume_np_loader", "consume_tf"))
      assertEquals(EnumSet.of(TensorOrigin.PARAMETER), sinkOrigins.parameterOrigins().get(sink));

    assertEquals(
        List.of(
            EnumSet.of(TensorOrigin.PARAMETER), // np_add(m, m): `a + b` over parameters
            EnumSet.of(TensorOrigin.NUMPY), // np_method(m): `m.reshape(...)`
            EnumSet.of(TensorOrigin.NUMPY), // make_array(): `np.array(...)` returned
            EnumSet.of(TensorOrigin.PARAMETER), // np_add3(m, m, m): nested operator over parameters
            EnumSet.of(TensorOrigin.PARAMETER)), // scale(m): scalar co-operand keeps `m`'s origin
        sinkOrigins.argumentOrigins().get("consume_np"));
    // The opaque value is an unknown-shape ndarray: provenance must survive even though the shape
    // does not resolve, since a consumer needs the origin most exactly when the type is ⊤.
    assertEquals(
        List.of(EnumSet.of(TensorOrigin.NUMPY)),
        sinkOrigins.argumentOrigins().get("consume_np_opaque"));
    // All six Keras dataset loaders route here: `load_data` returns ndarrays, so their results
    // are numpy-origin by the consumer-ratified runtime-type rule.
    assertEquals(
        nCopies(6, EnumSet.of(TensorOrigin.NUMPY)),
        sinkOrigins.argumentOrigins().get("consume_np_loader"));
    assertEquals(
        List.of(EnumSet.of(TensorOrigin.PARAMETER)),
        sinkOrigins.argumentOrigins().get("consume_tf"));
  }

  /**
   * The wala/ML#726 pins. The {@code f(x): return x + 1} distillation fed {@code np.ones(...)}
   * carries the hybridization-frame origin: under {@code tf.function} tracing the parameter is a
   * symbolic tensor and the {@code +} lowers to {@code tf.add}, so the eager numpy feed must not
   * classify the result numpy-only. The numpy-body counter-case (numpy literals, an operator over
   * locals, an interprocedural numpy return &mdash; no parameter provenance in any def) keeps
   * reading pure numpy, pinning the decline a consumer still needs. The local-level mixed operator
   * ({@code ndarray + Tensor}) pins runtime dispatch below the parameter boundary.
   *
   * @throws ClassHierarchyException If the class hierarchy cannot be built.
   * @throws CancelException If the analysis is canceled.
   * @throws IOException If the test file cannot be read.
   */
  @Test
  public void testParameterOrigin() throws ClassHierarchyException, CancelException, IOException {
    SinkOrigins sinkOrigins =
        getSinkOrigins("tf2_test_parameter_origin.py", "consume_param", "consume_np", "consume_tf");

    for (String sink : List.of("consume_param", "consume_np", "consume_tf"))
      assertEquals(EnumSet.of(TensorOrigin.PARAMETER), sinkOrigins.parameterOrigins().get(sink));

    assertEquals(
        List.of(EnumSet.of(TensorOrigin.PARAMETER)),
        sinkOrigins.argumentOrigins().get("consume_param"));
    assertEquals(
        List.of(EnumSet.of(TensorOrigin.NUMPY)), sinkOrigins.argumentOrigins().get("consume_np"));
    assertEquals(
        List.of(EnumSet.of(TensorOrigin.TENSORFLOW)),
        sinkOrigins.argumentOrigins().get("consume_tf"));
  }

  /**
   * Runs the tensor analysis on the given file and collects, for each named sink function, the
   * origins of its first parameter (value number 2, unioned across calling contexts) and of the
   * caller-side argument local at each of its call sites (in source order).
   *
   * @param filename The Python test file to analyze.
   * @param sinkFunctionNames The sink functions to observe.
   * @return The origins observed at the sinks.
   * @throws ClassHierarchyException If the class hierarchy cannot be built.
   * @throws CancelException If the analysis is canceled.
   * @throws IOException If the test file cannot be read.
   */
  private SinkOrigins getSinkOrigins(String filename, String... sinkFunctionNames)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine = makeEngine(emptyList(), filename);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();

    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);

    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    Map<PointerKey, Set<TensorOrigin>> origins = new HashMap<>();
    Map<String, Set<TensorOrigin>> parameterOrigins = new HashMap<>();

    analysis.forEach(
        pt -> {
          origins.put(pt.fst, pt.snd.getOrigins());

          if (!(pt.fst instanceof LocalPointerKey)) return;
          LocalPointerKey localPointerKey = (LocalPointerKey) pt.fst;

          // The first explicit parameter (value number 2; value number 1 is the function object).
          if (localPointerKey.getValueNumber() != 2) return;

          String signature = localPointerKey.getNode().getMethod().getSignature();

          for (String sink : sinkFunctionNames)
            // Signatures render scope separators as dots: `script f.py.consume_np.do()LRoot;`.
            if (signature.contains("." + sink + ".do("))
              parameterOrigins
                  .computeIfAbsent(sink, k -> EnumSet.noneOf(TensorOrigin.class))
                  .addAll(pt.snd.getOrigins());
        });

    // The caller-side argument local at each sink call site, keyed by instruction index for
    // source order and unioned across the caller's calling contexts.
    Map<String, SortedMap<Integer, Set<TensorOrigin>>> argumentsBySite = new HashMap<>();

    for (CGNode node : CG) {
      IR ir = node.getIR();
      if (ir == null) continue;

      SSAInstruction[] instructions = ir.getInstructions();

      for (int i = 0; i < instructions.length; i++) {
        if (!(instructions[i] instanceof SSAAbstractInvokeInstruction)) continue;
        SSAAbstractInvokeInstruction call = (SSAAbstractInvokeInstruction) instructions[i];

        for (CGNode target : CG.getPossibleTargets(node, call.getCallSite())) {
          String signature = target.getMethod().getSignature();

          for (String sink : sinkFunctionNames)
            if (signature.contains("." + sink + ".do(")) {
              // Use 0 is the callee function object; use 1 is the first positional argument.
              PointerKey argument =
                  builder
                      .getPointerAnalysis()
                      .getHeapModel()
                      .getPointerKeyForLocal(node, call.getUse(1));
              argumentsBySite
                  .computeIfAbsent(sink, k -> new TreeMap<>())
                  .computeIfAbsent(i, k -> EnumSet.noneOf(TensorOrigin.class))
                  .addAll(origins.getOrDefault(argument, EnumSet.noneOf(TensorOrigin.class)));
            }
        }
      }
    }

    Map<String, List<Set<TensorOrigin>>> argumentOrigins = new HashMap<>();
    argumentsBySite.forEach(
        (sink, bySite) -> argumentOrigins.put(sink, new ArrayList<>(bySite.values())));

    return new SinkOrigins(parameterOrigins, argumentOrigins);
  }
}
