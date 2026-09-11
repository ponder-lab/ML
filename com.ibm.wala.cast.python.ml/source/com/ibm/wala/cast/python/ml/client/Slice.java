package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for {@code tf.slice(input_, begin, size, name=None)}. Output dtype is inherited from
 * the {@code input_} input. Output shape is derived per axis from the constant {@code begin}/{@code
 * size} extents and {@code input_.shape}:
 *
 * <pre>
 * output.shape[i] = size[i]                    if size[i] &gt;= 0
 *                 = input_.shape[i] - begin[i] if size[i] == -1  ("all remaining")
 * </pre>
 *
 * A constant {@code size} with no {@code -1} entries gives a fully concrete shape independent of
 * {@code input_.shape}; a {@code -1} entry needs the corresponding {@code input_} dim and {@code
 * begin[i]}, and degrades on that axis (keeping the rank) when either is non-constant. Bounds that
 * do not resolve at all degrade the same way rather than losing the shape: a slice never changes
 * the rank at run time, so an unresolvable {@code size} keeps the input's rank with every extent
 * degraded per the wala/ML#721 conventions, and an unresolvable {@code begin} affects only the
 * {@code size}-of-{@code -1} axes. The shape falls back to ⊤ only when the resolved ranks disagree
 * with {@code input_} or a resolved {@code size} entry is invalid. See <a
 * href="https://github.com/wala/ML/issues/569">wala/ML#569</a>; dtype forwarding alone landed in <a
 * href="https://github.com/wala/ML/issues/568">wala/ML#568</a>.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/slice">tf.slice</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Slice extends PassThroughUnaryTensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(Slice.class.getName());

  public Slice(PointsToSetVariable source) {
    super(source);
  }

  public Slice(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "input_";
  }

  /**
   * Derives the output shape per axis from the constant {@code begin} (arg 1) and {@code size} (arg
   * 2) extents together with the {@code input_} (arg 0) shape, per the rule documented on the
   * class.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when {@code input_}'s shape is
   *     unknown, a resolved rank disagrees with {@code input_}, or a resolved {@code size} entry is
   *     invalid; unresolvable bounds degrade the extents while the rank survives.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // input_ is arg 0 (resolved by the passthrough base); begin is arg 1; size is arg 2.
    Set<List<Dimension<?>>> inputShapes = super.getDefaultShapes(builder);
    if (inputShapes == null) return null;

    ShapeTransform rule = this.sliceRule(builder);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> input : inputShapes) {
      Set<List<Dimension<?>>> outs = rule.apply(input);
      // A ⊤ (null) for any combination joins to ⊤ for the whole result: returning only the
      // concrete subset would under-approximate the possible shapes.
      if (outs == null) return null;
      ret.addAll(outs);
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * The {@code tf.slice} shape rule as a function of the {@code input_} shape alone, its {@code
   * begin} (arg 1) and {@code size} (arg 2) bounds resolved once from the points-to substrate. One
   * rule serves both arms: {@link #getDefaultShapes} applies it to the input's substrate shape and
   * the {@link TypeFeedKind#TRANSFORM} feed applies it to the input's dataflow state, so an input
   * typed only by dataflow (a sidecar annotation, wala/ML#905) gets the same answer as one the
   * substrate resolves.
   *
   * <p>A slice never changes the rank at run time, so bounds that do not resolve degrade the
   * extents and not the shape: the output keeps the input's rank with every axis degraded per the
   * wala/ML#721 conventions. The historical whole-shape ⊤ there dropped the rank of every value
   * downstream of a runtime-computed crop (a {@code tf.slice} over {@code
   * tf.image.sample_distorted_bounding_box}'s outputs), which is how an augmentation chain's
   * helpers lost their image parameters' ranks. The documented-contract crop keeps its channel
   * extent (wala/ML#844): when both bounds are the destructured outputs of one {@code
   * sample_distorted_bounding_box} call, {@code size} is documented as {@code [target_height,
   * target_width, -1]} and {@code begin} as {@code [offset_height, offset_width, 0]}, so a rank-3
   * input's channel extent survives verbatim while the spatial axes stay degraded.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The rule; it yields {@code null} for an input whose ranks disagree with a resolved
   *     bound or whose resolved {@code size} entry is invalid.
   */
  private ShapeTransform sliceRule(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> beginLists = resolveConstantIntList(builder, 1, "begin");
    Set<List<Dimension<?>>> sizeLists = resolveConstantIntList(builder, 2, "size");

    if (sizeLists == null) {
      boolean cropContract = this.boundsAreSampleDistortedBoundingBoxOutputs(builder);
      return input -> {
        List<Dimension<?>> out = new ArrayList<>(input.size());
        for (int i = 0; i < input.size(); i++) {
          Dimension<?> inDim = input.get(i);
          if (cropContract && input.size() == 3 && i == 2) out.add(inDim);
          else out.add(inDim instanceof DynamicDim ? DynamicDim.INSTANCE : UnresolvedDim.INSTANCE);
        }
        return Set.of(out);
      };
    }

    return input -> {
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (List<Dimension<?>> size : sizeLists)
        // An unresolvable `begin` alone does not lose the shape: only the size-of--1 axes need it,
        // and those degrade per axis inside the rule.
        for (List<Dimension<?>> begin :
            beginLists == null ? Collections.<List<Dimension<?>>>singleton(null) : beginLists) {
          List<Dimension<?>> out = sliceShape(input, begin, size);
          if (out == null) return null;
          ret.add(out);
        }
      return ret;
    };
  }

  /** The producer whose bounds carry the documented crop contract (wala/ML#844). */
  private static final String CROP_PRODUCER_NAME = "sample_distorted_bounding_box";

  /**
   * Whether every invocation anchoring this generator takes its {@code begin} and {@code size} from
   * elements 0 and 1 of the same {@code sample_distorted_bounding_box} result (wala/ML#844).
   *
   * <p>The match is deliberately narrow, since firing on a different producer that happens to share
   * the destructuring shape would fabricate an extent: both bounds must be property reads of the
   * constant members 0 and 1 on one tuple, and that tuple's producing invoke's callee must be an
   * attribute read of exactly this operation's name. Every candidate invocation must match, so a
   * manual anchor whose callers disagree keeps the all-degraded answer. The failure direction is
   * silence: a pattern this does not recognize degrades to unresolved extents, never to a claimed
   * one.
   *
   * @param builder The propagation call graph builder.
   * @return {@code true} iff the documented crop contract applies to every anchoring invocation.
   */
  private boolean boundsAreSampleDistortedBoundingBoxOutputs(PropagationCallGraphBuilder builder) {
    PythonInvokeInstruction call = this.getInvokeInstruction();
    if (call != null) return isCropContractCall(this.getNode(), call);

    boolean any = false;
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) return false;
      if (!isCropContractCall(callerInvoke.fst, (PythonInvokeInstruction) callerInvoke.snd))
        return false;
      any = true;
    }
    return any;
  }

  /**
   * The per-invocation half of {@link #boundsAreSampleDistortedBoundingBoxOutputs}: whether one
   * {@code tf.slice} call's {@code begin} and {@code size} are the destructured first and second
   * elements of a single {@code sample_distorted_bounding_box} result.
   *
   * @param node The calling frame.
   * @param call The {@code tf.slice} invocation.
   * @return {@code true} iff the documented crop contract applies to this invocation.
   */
  private static boolean isCropContractCall(CGNode node, PythonInvokeInstruction call) {
    if (node.getIR() == null) return false;
    SymbolTable st = node.getIR().getSymbolTable();
    DefUse du = node.getDU();

    int beginVn = call.getUse("begin");
    if (beginVn == -1 && call.getNumberOfPositionalParameters() > 2) beginVn = call.getUse(2);
    int sizeVn = call.getUse("size");
    if (sizeVn == -1 && call.getNumberOfPositionalParameters() > 3) sizeVn = call.getUse(3);
    if (beginVn <= 0 || sizeVn <= 0) return false;

    SSAInstruction beginDef = du.getDef(beginVn);
    SSAInstruction sizeDef = du.getDef(sizeVn);
    if (!(beginDef instanceof PythonPropertyRead) || !(sizeDef instanceof PythonPropertyRead))
      return false;
    PythonPropertyRead beginRead = (PythonPropertyRead) beginDef;
    PythonPropertyRead sizeRead = (PythonPropertyRead) sizeDef;
    if (beginRead.getObjectRef() != sizeRead.getObjectRef()) return false;
    if (!isConstantMember(st, beginRead.getMemberRef(), 0)
        || !isConstantMember(st, sizeRead.getMemberRef(), 1)) return false;

    SSAInstruction tupleDef = du.getDef(beginRead.getObjectRef());
    if (!(tupleDef instanceof PythonInvokeInstruction)) return false;
    SSAInstruction calleeDef = du.getDef(((PythonInvokeInstruction) tupleDef).getUse(0));
    if (!(calleeDef instanceof PythonPropertyRead)) return false;
    int calleeMemberVn = ((PythonPropertyRead) calleeDef).getMemberRef();
    return st.isStringConstant(calleeMemberVn)
        && CROP_PRODUCER_NAME.equals(st.getStringValue(calleeMemberVn));
  }

  /**
   * Whether the given value number is a constant equal to the expected tuple-field index.
   *
   * @param st The frame's symbol table.
   * @param memberVn The member reference's value number.
   * @param expected The expected field index.
   * @return {@code true} iff the member is that constant.
   */
  private static boolean isConstantMember(SymbolTable st, int memberVn, int expected) {
    if (!st.isConstant(memberVn)) return false;
    Object value = st.getConstantValue(memberVn);
    return value != null && String.valueOf(expected).equals(String.valueOf(value));
  }

  /**
   * Resolves a {@code begin}/{@code size} argument (a constant list/tuple or {@code tf.constant} of
   * ints) into its dimension lists via {@link #getShapesFromShapeArgument}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param position The 0-based positional index of the argument (excluding {@code self}).
   * @param name The keyword name of the argument.
   * @return The resolved int-list candidates, or {@code null} when the argument's points-to set is
   *     empty or it cannot be resolved to a constant list.
   */
  private Set<List<Dimension<?>>> resolveConstantIntList(
      PropagationCallGraphBuilder builder, int position, String name) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, position, name);
    if (pts == null || pts.isEmpty()) return null;
    try {
      Set<List<Dimension<?>>> lists = this.getShapesFromShapeArgument(builder, pts);
      return (lists == null || lists.isEmpty()) ? null : lists;
    } catch (IllegalStateException e) {
      // `getShapesFromShapeArgument` throws `IllegalStateException` for an unrecognized shape form;
      // degrade that to ⊤. Other runtime exceptions propagate as intended diagnostics.
      LOGGER.fine(
          () -> "Could not resolve " + name + " of " + describe(this.getSource()) + ": " + e + ".");
      return null;
    }
  }

  /**
   * Applies the {@code tf.slice} per-axis shape rule to a single {@code (input, begin, size)}
   * combination.
   *
   * @param input The {@code input_} shape.
   * @param begin The constant {@code begin} offsets, or {@code null} when {@code begin} did not
   *     resolve; only the {@code size}-of-{@code -1} axes need it, and those degrade per axis.
   * @param size The constant {@code size} extents.
   * @return The output shape, or {@code null} (⊤) when the ranks disagree, a {@code size} entry is
   *     invalid ({@code < -1}), or a {@code size}-of-{@code -1} axis computes a negative extent
   *     ({@code begin} past the axis).
   */
  private static List<Dimension<?>> sliceShape(
      List<Dimension<?>> input, List<Dimension<?>> begin, List<Dimension<?>> size) {
    int rank = input.size();
    if ((begin != null && begin.size() != rank) || size.size() != rank) return null;

    List<Dimension<?>> out = new ArrayList<>(rank);
    for (int i = 0; i < rank; i++) {
      Dimension<?> sizeDim = size.get(i);
      if (!(sizeDim instanceof NumericDim)) {
        // An unresolvable extent inside an otherwise-resolved size list degrades this axis and not
        // the shape: the rank is the size list's own length (wala/ML#721).
        out.add(sizeDim instanceof DynamicDim ? DynamicDim.INSTANCE : UnresolvedDim.INSTANCE);
        continue;
      }
      int s = ((NumericDim) sizeDim).value();
      if (s >= 0) {
        out.add(new NumericDim(s));
      } else if (s == -1) {
        // "all remaining" along axis i: input_.shape[i] - begin[i], when both are constant.
        Dimension<?> inDim = input.get(i);
        Dimension<?> beginDim = begin == null ? null : begin.get(i);
        if (inDim instanceof NumericDim && beginDim instanceof NumericDim) {
          int extent = ((NumericDim) inDim).value() - ((NumericDim) beginDim).value();
          // A `begin` past the axis would yield a negative (invalid) extent; degrade to ⊤.
          if (extent < 0) return null;
          out.add(new NumericDim(extent));
        } else if (beginDim instanceof NumericDim && ((NumericDim) beginDim).value() == 0)
          // Taking the axis in full from offset zero leaves the extent exactly as it arrived, so
          // the dimension carries through verbatim whatever kind it has (wala/ML#899). Degrading
          // it here relabelled a `Symbolic` reshape placeholder as `Unresolved`, which asserts a
          // fixed runtime size the placeholder never claimed, and which `mergeAnnotationDims`
          // treats as an extent an annotation may fill. `SliceBuiltinOperation.sliceExtent`
          // already returns the receiver's dimension untouched for the equivalent bare `:`.
          out.add(inDim);
        else
          // Keep the rank. The remaining extent of a `None` axis is itself `None` at run time;
          // otherwise it is a fixed size the analysis could not compute (wala/ML#721).
          out.add(inDim instanceof DynamicDim ? DynamicDim.INSTANCE : UnresolvedDim.INSTANCE);
      } else {
        return null; // size < -1 is invalid for tf.slice.
      }
    }
    return out;
  }

  /**
   * Collapse-safe record view (wala/ML#718): this generator transforms its input shapes in {@link
   * #getDefaultShapes}, which the pass-through identity record path would bypass, so the record
   * view routes through the legacy transform until a member-wise upgrade.
   *
   * @param builder The propagation call graph builder.
   * @return The transformed result, with any partial input collapsed by the legacy view.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    return ShapeResult.fromLegacy(this.getDefaultShapes(builder));
  }

  /**
   * The feed carries this generator's own shape rule (wala/ML#905), so an input typed only by
   * dataflow gets exactly the answer the substrate arm computes: the fully-taken channel of the
   * documented crop survives, and constant bounds slice the fed shape as they slice a resolved one.
   * The former {@link TypeFeedKind#RANK_PRESERVING} declaration degraded every fed axis, which is
   * how an annotated image lost its channel through a crop that keeps it.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The rule-carrying feed over the caller-side input keys, or {@code null} when none is
   *     located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return this.getTypeFeed(builder, this.sliceRule(builder));
  }
}
