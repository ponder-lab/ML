package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_16_28_28_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_32_28_28_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_8_FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.COMPLEX64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
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

/**
 * Cross-cutting analysis-mechanics tests (value numbering, recursion, containers, entrypoints,
 * strictness) without a feature-area home yet, carved from the {@code TestTensorflow2Model}
 * monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestMisc extends AbstractTensorTest {

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
}
