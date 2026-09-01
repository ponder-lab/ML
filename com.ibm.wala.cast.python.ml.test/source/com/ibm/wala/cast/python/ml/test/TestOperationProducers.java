package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * The {@code Operation} producers (<a
 * href="https://github.com/wala/ML/issues/864">wala/ML#864</a>).
 *
 * <p>TensorFlow rejects an {@code Operation} as the return of a traced function, so a consumer
 * deciding whether a function can be traced must identify one POSITIVELY. Absence of a tensor type
 * cannot serve, because a function returning a Python integer or {@code None} is equally absent.
 * That is why this asserts on the ALLOCATION rather than on a type.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestOperationProducers extends TestPythonMLCallGraphShape {

  /** The type an {@code Operation}-producing API allocates. */
  private static final String OPERATION = "Ltensorflow/python/framework/ops/Operation";

  /** The type its tensor sibling allocates, for the control. */
  private static final String TENSOR = "Ltensorflow/python/framework/ops/Tensor";

  /**
   * A function returning an operation has exactly that allocation in its RETURN value's points-to
   * set, and a sibling returning a tensor has the tensor's.
   *
   * <p>This is the query a consumer actually makes. Deciding whether a function can be traced means
   * asking what its RETURN holds, so asserting merely that an operation is allocated somewhere in
   * the heap would leave the useful question untested. That weaker assertion is not hypothetically
   * weak: {@code matmul} has long allocated an operation to fill a tensor's {@code op} field, so a
   * heap-wide check can be satisfied by an allocation that reaches no return value at all.
   *
   * <p>The pair is the point. Absence of a tensor type cannot discriminate, since a function
   * returning a Python integer or {@code None} is equally absent; these two differ in what they
   * positively hold.
   *
   * <p>A consumer's predicate should require EVERY member to be an operation rather than any,
   * because a value that may hold an operation or a tensor depending on the path is not one the
   * analysis can decide, and deciding it either way is worse than declining. Note also that one
   * function can appear as several context-sensitive nodes, so the query unions over them.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testReturnedOperationIsVisibleInThePointsToSet() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_operation.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);

    assertEquals(
        "`tf.group`'s caller returns an operation.",
        Set.of(OPERATION),
        returnedAllocations(builder, cg, "returns_operation"));
    assertEquals(
        "`tf.no_op`'s caller returns an operation.",
        Set.of(OPERATION),
        returnedAllocations(builder, cg, "returns_no_op"));
    assertEquals(
        "The control returns a tensor, so the two are positively distinguishable.",
        Set.of(TENSOR),
        returnedAllocations(builder, cg, "returns_tensor"));
  }

  /**
   * The CONDITIONAL producers resolve to an operation too, under the traced reading (wala/ML#864).
   *
   * <p>These differ in kind from the pair above, and the difference is recorded rather than
   * smoothed over. Eagerly they evaluate to {@code None}; only under tracing do they evaluate to an
   * operation, and the fixture's own asserts pin the eager behaviour so the approximation is
   * visible next to the model rather than only in a comment.
   *
   * <p>The model states the traced reading because that is the only reading anyone queries: the
   * question asked of operation identity is whether a function can be traced, which is always about
   * the traced context. A consumer asking what these evaluate to EAGERLY would be told something
   * false, and the honest response to that is a traced context distinction rather than a different
   * allocation here.
   *
   * <p>The keyword spellings are asserted because they are what real code writes, NOT because the
   * summaries' parameter lists make or break them. Measured both ways: with the parameter lists
   * enumerated as they were, and with them corrected, every one of these still resolves. A summary
   * whose body is {@code <new>} plus {@code <return>} reads no parameter, so the returned
   * allocation cannot depend on how arguments bind; a parameter list matters to a summary that USES
   * one, such as a pass-through return or a generator reading an argument's points-to set. What
   * these assertions pin is coverage of the spellings a consumer will meet, which is worth having
   * on its own and is not a witness for the parameter lists.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testConditionalProducersResolveUnderTheTracedReading() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_operation.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);

    assertEquals(
        "`tf.print` returns an operation under tracing.",
        Set.of(OPERATION),
        returnedAllocations(builder, cg, "returns_print"));
    assertEquals(
        "`tf.assert_equal` returns an operation under tracing.",
        Set.of(OPERATION),
        returnedAllocations(builder, cg, "returns_assert"));
    assertEquals(
        "A keyword spelling resolves too.",
        Set.of(OPERATION),
        returnedAllocations(builder, cg, "returns_assert_named"));
    assertEquals(
        "So does a keyword the summary does not name, in the spelling real code uses.",
        Set.of(OPERATION),
        returnedAllocations(builder, cg, "returns_print_kwarg"));
    assertEquals(
        "And the variadic unconditional producer, for the same reason.",
        Set.of(OPERATION),
        returnedAllocations(builder, cg, "returns_group_kwarg"));
  }

  /**
   * The allocation types a named function's return value may hold, unioned over its context-
   * sensitive nodes.
   *
   * @param builder The call graph builder.
   * @param cg The call graph.
   * @param function The function's name.
   * @return The type names, with any non-allocation member recorded rather than dropped.
   */
  private static Set<String> returnedAllocations(
      PythonSSAPropagationCallGraphBuilder builder, CallGraph cg, String function) {
    Set<String> ret = new HashSet<>();

    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().contains("." + function + ".")) continue;

      PointerKey returnKey =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForReturnValue(node);

      for (InstanceKey key : builder.getPointerAnalysis().getPointsToSet(returnKey))
        // A member that is not an allocation is RECORDED rather than skipped, so the equality
        // assertion fails on it. Skipping would let a set holding an operation beside a constant
        // key read as {OPERATION}, which is the every-member rule this test exists to pin being
        // broken by the test itself. It is also the same skip-versus-record mistake the dtype
        // resolver was corrected for in wala/ML#860: a dropped member makes a mixed set look pure.
        ret.add(
            key instanceof AllocationSiteInNode allocation
                ? allocation.getSite().getDeclaredType().getName().toString()
                : "non-allocation:" + key.getClass().getSimpleName());
    }

    return ret;
  }
}
