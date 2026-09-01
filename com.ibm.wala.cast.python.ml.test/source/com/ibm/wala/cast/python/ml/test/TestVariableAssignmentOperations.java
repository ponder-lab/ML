package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * Reading {@code .op} on a variable assignment yields an {@code Operation} (a follow-on to <a
 * href="https://github.com/wala/ML/issues/864">wala/ML#864</a>).
 *
 * <p>The assignment family was previously absent from the model file entirely, so the RECEIVER of a
 * {@code .op} read resolved to nothing. That is a different defect from a missing {@code op} field
 * on a resolved receiver, and the distinction matters because the two are indistinguishable from a
 * consumer's side while having different fixes. The assertions below separate them.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestVariableAssignmentOperations extends TestPythonMLCallGraphShape {

  /** The type a {@code .op} read resolves to. */
  private static final String OPERATION = "Ltensorflow/python/framework/ops/Operation";

  /** The marker the helper records for a member that is not an allocation. */
  private static final String NON_ALLOCATION_PREFIX = "non-allocation:";

  /** The type the assignment itself resolves to. */
  private static final String TENSOR = "Ltensorflow/python/framework/ops/Tensor";

  /**
   * The assignment resolves, and its {@code op} resolves to an operation.
   *
   * <p>The pair is the point, and it is not decoration. An unresolved receiver and a resolved
   * receiver missing its {@code op} field both leave the {@code .op} read empty, so asserting only
   * the operation would not say which defect had been fixed, nor notice if a later change traded
   * one for the other. The first assertion pins that the receiver exists; the second pins what its
   * field holds.
   *
   * <p>The reproduction is used verbatim rather than paraphrased. A reduction that varies something
   * other than what the real case differs in passes while the real case fails, so the spelling here
   * is the one actually reported: {@code v.assign_add(1.0).op} on a scalar variable.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testAssignmentAndItsOperationBothResolve() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_variable_operation.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);

    assertEquals(
        "The assignment itself resolves, so an empty `.op` read would mean a missing field rather"
            + " than an unresolved receiver.",
        Set.of(TENSOR),
        returnedAllocations(builder, cg, "returns_assignment"));
    assertEquals(
        "And its `op` resolves to an operation, under the traced reading.",
        Set.of(OPERATION),
        returnedAllocations(builder, cg, "returns_assignment_op"));
  }

  /**
   * Every member of the family resolves, not merely the reported one.
   *
   * <p>The six are an enumeration rather than a mechanism, so each needs its own assertion: a
   * mechanism can be witnessed once, but a list can be wrong entry by entry, and an entry with no
   * assertion borrows verification from its neighbours. All six were confirmed against the runtime,
   * where each exists, each result carries {@code .op}, and returning one from a traced function
   * raises.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testEveryMemberOfTheFamilyResolves() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_variable_operation.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);

    for (String function :
        new String[] {
          "returns_assign_op",
          "returns_assign_sub_op",
          "returns_scatter_add_op",
          "returns_scatter_sub_op",
          "returns_scatter_update_op"
        })
      assertEquals(
          "`" + function + "` resolves to an operation.",
          Set.of(OPERATION),
          returnedAllocations(builder, cg, function));
  }

  /**
   * A member that is not an allocation is RECORDED rather than dropped.
   *
   * <p>The helper's contract is that a mixed points-to set cannot read as a pure one, since
   * silently skipping a member would let a set holding an operation beside something else satisfy
   * an equality assertion for {@code {OPERATION}}. That is the every-member rule these tests exist
   * to pin, and until now it was only asserted in prose: every fixture function returned an
   * allocation, so the recording branch never ran and the claim had no witness.
   *
   * <p>A function returning a Python literal supplies one. Its return resolves to a {@code
   * ConstantKey}, which is not an {@code AllocationSiteInNode}, so the branch executes and its
   * marker is observable. Were the helper to skip instead of record, this would read as the empty
   * set.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testNonAllocationMembersAreRecordedRatherThanDropped() throws Exception {
    PythonTensorAnalysisEngine engine =
        (PythonTensorAnalysisEngine) makeEngine("tf2_test_variable_operation.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    assertNotNull(cg);

    assertEquals(
        "A literal's return is a constant rather than an allocation, and the helper records it. An"
            + " empty set here would mean the member had been silently dropped.",
        Set.of(NON_ALLOCATION_PREFIX + ConstantKey.class.getSimpleName()),
        returnedAllocations(builder, cg, "returns_literal"));
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
        // Recorded rather than skipped, so a mixed set cannot read as a pure one.
        ret.add(
            key instanceof AllocationSiteInNode allocation
                ? allocation.getSite().getDeclaredType().getName().toString()
                : NON_ALLOCATION_PREFIX + key.getClass().getSimpleName());
    }

    return ret;
  }
}
