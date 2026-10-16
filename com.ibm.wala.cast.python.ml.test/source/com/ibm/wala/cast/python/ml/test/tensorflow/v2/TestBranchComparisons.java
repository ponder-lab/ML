package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptySet;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import com.ibm.wala.cast.ir.ssa.AstAssertInstruction;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.SSAConditionalBranchInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;
import org.junit.Test;

/**
 * Tests that a comparison whose only use is a condition is typed (<a
 * href="https://github.com/wala/ML/issues/1074">wala/ML#1074</a>).
 *
 * <p>A comparison passed to a call takes part in a points-to constraint, which makes it a node of
 * the tensor dataflow graph, but one that only an {@code if}, a {@code while} or an {@code assert}
 * tests does not. So each test reads the type of the value a function's first conditional branch
 * tests, rather than of a sink's parameter, which would itself put the comparison in the graph.
 */
public class TestBranchComparisons extends AbstractTensorTest {

  private static final Logger LOGGER = Logger.getLogger(TestBranchComparisons.class.getName());

  private static final String FILE = "tf2_test_branch_comparisons.py";

  /**
   * The types of the values the conditional branches and assertions test, by function name and in
   * instruction order, from one analysis of {@link #FILE}.
   */
  private static Map<String, List<Set<TensorType>>> branchConditionTypes;

  /**
   * Returns, for each conditional branch or assertion in the named function in instruction order,
   * the union over contexts of the types of the value it tests.
   *
   * @param functionName The function's name, qualified by its enclosing functions.
   * @return The types per branch, each empty when the value is not a tensor.
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  private List<Set<TensorType>> branchConditionTypes(String functionName)
      throws ClassHierarchyException, CancelException, IOException {
    if (branchConditionTypes == null) {
      PythonTensorAnalysisEngine engine =
          makeEngine(
              PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH,
              this.getPathFiles(null),
              new String[] {FILE});
      PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
      CallGraph callGraph = builder.makeCallGraph(builder.getOptions());
      TensorTypeAnalysis analysis = engine.performAnalysis(builder);

      // Indexed by node and value number, as a consumer reading a branch's operand looks it up.
      Map<CGNode, Map<Integer, Set<TensorType>>> types = new HashMap<>();
      analysis.forEach(
          pair -> {
            if (pair.fst instanceof LocalPointerKey key && pair.snd.getTypes() != null)
              types
                  .computeIfAbsent(key.getNode(), n -> new HashMap<>())
                  .computeIfAbsent(key.getValueNumber(), v -> new HashSet<>())
                  .addAll(pair.snd.getTypes());
          });

      Map<String, List<Set<TensorType>>> result = new HashMap<>();
      String prefix = "script " + FILE + ".";
      String suffix = ".do()LRoot;";
      for (CGNode node : callGraph) {
        String signature = node.getMethod().getSignature();
        // The script's own body is named by the prefix alone and holds no function.
        if (!signature.startsWith(prefix)
            || !signature.endsWith(suffix)
            || signature.length() <= prefix.length() + suffix.length()) continue;
        String name = signature.substring(prefix.length(), signature.length() - suffix.length());
        List<Set<TensorType>> branches = new ArrayList<>();
        for (SSAInstruction instruction : node.getIR().getInstructions()) {
          if (instruction instanceof SSAConditionalBranchInstruction
              || instruction instanceof AstAssertInstruction)
            branches.add(
                new HashSet<>(
                    types
                        .getOrDefault(node, Map.of())
                        .getOrDefault(instruction.getUse(0), emptySet())));
        }
        result.merge(
            name,
            branches,
            (a, b) -> {
              for (int i = 0; i < a.size() && i < b.size(); i++) a.get(i).addAll(b.get(i));
              return a;
            });
      }
      branchConditionTypes = result;
      LOGGER.fine(() -> "Branch condition types: " + result + ".");
    }
    List<Set<TensorType>> branches = branchConditionTypes.get(functionName);
    assertFalse(
        "Function "
            + functionName
            + " must have a condition in the call graph; found "
            + branchConditionTypes.keySet()
            + ".",
        branches == null || branches.isEmpty());
    return branches;
  }

  /**
   * Returns the types of the value the first conditional branch or assertion in the named function
   * tests.
   *
   * @param functionName The function's name, qualified by its enclosing functions.
   * @return The types, empty when the value is not a tensor.
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  private Set<TensorType> conditionTypes(String functionName)
      throws ClassHierarchyException, CancelException, IOException {
    return branchConditionTypes(functionName).get(0);
  }

  /**
   * {@code if tf.reduce_sum(x) > 0.0: return None} tests a scalar {@code bool}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIfGreater() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(SCALAR_TENSOR_OF_BOOL), conditionTypes("if_greater"));
  }

  /**
   * {@code if tf.reduce_sum(x) <= 0.0:} with no {@code else}, merging {@code None} with a tensor,
   * tests a scalar {@code bool}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIfLessEqual() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(SCALAR_TENSOR_OF_BOOL), conditionTypes("if_less_equal"));
  }

  /**
   * {@code while tf.reduce_sum(x) > 0.0:} tests a scalar {@code bool}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testWhileGreater() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(SCALAR_TENSOR_OF_BOOL), conditionTypes("while_greater"));
  }

  /**
   * {@code assert tf.reduce_sum(x) > 0.0} tests a scalar {@code bool}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testAssertGreater() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(SCALAR_TENSOR_OF_BOOL), conditionTypes("assert_greater"));
  }

  /**
   * A nested function's {@code if} on a comparison of the enclosing function's tensor tests a
   * scalar {@code bool}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNestedFunction() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(SCALAR_TENSOR_OF_BOOL), conditionTypes("nested.inner"));
  }

  /**
   * A {@code tf.while_loop} body's {@code if} on a comparison of a loop variable tests a scalar
   * {@code bool}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testWhileLoopBody() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(SCALAR_TENSOR_OF_BOOL), conditionTypes("loop_body"));
  }

  /**
   * {@code if n > 0} on a Python {@code int} tests no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIfInt() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(emptySet(), conditionTypes("if_int"));
  }

  /**
   * {@code if len(xs) > 0} tests no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIfLen() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(emptySet(), conditionTypes("if_len"));
  }

  /**
   * {@code if x is None} on a tensor tests no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIfIsNone() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(emptySet(), conditionTypes("if_is_none"));
  }

  /**
   * {@code assert x.shape == (2, 3)} tests no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testAssertShape() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(emptySet(), conditionTypes("assert_shape"));
  }

  /**
   * {@code assert x.dtype == tf.float32} tests no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testAssertDtype() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(emptySet(), conditionTypes("assert_dtype"));
  }

  /**
   * {@code if x.shape[0] == 2} tests no tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIfDimension() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(emptySet(), conditionTypes("if_dimension"));
  }

  /**
   * Both conditions of {@code if a > 0.0 and b < 2.0:} test a scalar {@code bool}: the first tests
   * the left comparison, and the second the φ of the operand {@code and} selects.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIfAnd() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(
        List.of(Set.of(SCALAR_TENSOR_OF_BOOL), Set.of(SCALAR_TENSOR_OF_BOOL)),
        branchConditionTypes("if_and"));
  }

  /**
   * Both conditions of the chained {@code if 0.0 < s < 100.0:} test a scalar {@code bool}, whatever
   * the lowering's disjunction (wala/ML#1079) selects.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIfChained() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(
        List.of(Set.of(SCALAR_TENSOR_OF_BOOL), Set.of(SCALAR_TENSOR_OF_BOOL)),
        branchConditionTypes("if_chained"));
  }

  /**
   * {@code u if t > 0.0 else v} tests a scalar {@code bool}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testTernary() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(SCALAR_TENSOR_OF_BOOL), conditionTypes("ternary"));
  }

  /**
   * {@code if not (t > 0.0):} tests the negation of a comparison, which is not typed. At run time
   * {@code not} of a tensor is a Python {@code bool}, while AutoGraph converts it to {@code
   * tf.logical_not}, a {@code bool} tensor, under tracing, so which to answer is open.
   *
   * <p>TODO: Decide the negation's type under <a
   * href="https://github.com/wala/ML/issues/1074">wala/ML#1074</a>.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testIfNot() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(emptySet(), conditionTypes("if_not"));
  }

  /**
   * A comprehension's filter {@code if y > 0.0} lowers to a function returning the comparison, not
   * to a branch, and the returned comparison is typed through its return: the filter's parameter is
   * a scalar {@code float32} and its one local tensor the {@code bool} comparison.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testComprehensionFilter()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "comprehension_filter.filter2", 1, 2, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }
}
