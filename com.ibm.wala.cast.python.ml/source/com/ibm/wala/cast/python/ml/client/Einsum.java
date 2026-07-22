package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.logging.Logger;

/**
 * Generator for {@code tf.einsum(equation, *inputs, **kwargs)}. Parses the equation string (e.g.
 * {@code "ij,jk->ik"}) and composes the output shape from each input's shape: every output label is
 * resolved to the input dimension it names. Output dtype inherits from the first input whose dtype
 * resolves (the runtime requires the inputs' dtypes to agree, promotion happening upstream of the
 * call; wala/ML#737).
 *
 * <p>Models broadcasting ellipsis ({@code ...}, wala/ML#705): each input's ellipsis binds the axes
 * its letters don't consume, the per-input groups broadcast right-aligned, and the output's {@code
 * ...} receives the broadcast result. In implicit (arrow-less) mode the broadcast group precedes
 * the once-occurring labels. Repeated labels within one term (diagonal/trace forms such as {@code
 * "ii->i"} and {@code "ii"}) constrain the named axes equal and contribute a single output
 * dimension.
 *
 * <p>An input whose shape does not resolve no longer forces a ⊤ output (wala/ML#737): the equation
 * still proves the output rank and every axis a resolved operand binds, with the remaining axes
 * {@link UnresolvedDim}. The ⊤ fallback remains for an unresolved or malformed equation, an
 * unsatisfiable combination, and an output broadcast group that depends on an unresolved operand.
 *
 * <p>Argument layout at the call site: the {@code *inputs} varargs are spread positionally after
 * the equation (the packing into a single {@code inputs} tuple is callee-side only), so:
 *
 * <ul>
 *   <li>position 0: {@code equation} (string).
 *   <li>position 1: the first tensor input; the dtype source.
 *   <li>position 2, 3, ...: the remaining tensor inputs; input {@code i} of the equation's terms
 *       sits at position {@code i + 1}.
 * </ul>
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/linalg/einsum">tf.linalg.einsum</a>
 * @see <a href="https://github.com/wala/ML/issues/507">wala/ML#507</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Einsum extends PassThroughUnaryTensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(Einsum.class.getName());

  /**
   * Parameter positions and keyword names for {@code tf.einsum(equation, *inputs, **kwargs)}.
   * Ordinals match the position in {@code tensorflow.xml}'s {@code paramNames} after the implicit
   * {@code self} receiver. Note that {@code equation} is a string (not a tensor), so the dtype
   * source is {@code INPUTS} at position 1.
   */
  protected enum Parameters {
    /** The einsum equation string (e.g. {@code "ij,jk->ik"}). */
    EQUATION,

    /** The first tensor input (positional index 1); the dtype source and the first shape input. */
    INPUTS;

    /**
     * Lowercase keyword name used in argument-resolution helpers when the call site uses {@code
     * keyword=value} syntax.
     *
     * @return The lowercased enum name (e.g. {@code "inputs"}).
     */
    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    /**
     * Positional index of this parameter, excluding the implicit {@code self} receiver.
     *
     * @return The zero-based positional index.
     */
    public int getIndex() {
      return ordinal();
    }
  }

  public Einsum(PointsToSetVariable source) {
    super(source);
  }

  public Einsum(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return Parameters.INPUTS.getIndex();
  }

  @Override
  protected String getInputParameterName() {
    return Parameters.INPUTS.getName();
  }

  /**
   * The runtime requires einsum's inputs to agree on dtype (promotion happens upstream of the
   * call), so any resolved input proves the output dtype: consult each input in order and return
   * the first that resolves, rather than pinning to the first input alone (wala/ML#737; the
   * either-operand rationale of the {@code MatMul} dtype fix). Falls back to {@link DType#UNKNOWN}
   * when no input resolves or the equation's input count is itself unknown.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The output dtypes.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    String equation = this.getEquation(builder);
    ParsedEquation parsed = equation == null ? null : parseEquation(equation);
    int inputCount = parsed == null ? 1 : parsed.inputs().size();
    for (int i = 0; i < inputCount; i++) {
      Set<DType> dtypes = this.dtypesOfArg(builder, Parameters.INPUTS.getIndex() + i, null);
      if (dtypes != null && !dtypes.isEmpty() && !dtypes.contains(DType.UNKNOWN)) return dtypes;
    }
    return EnumSet.of(DType.UNKNOWN);
  }

  /**
   * The cross-product cap for multi-shape inputs (wala/ML#737): beyond it, multi-shape inputs
   * degrade to unresolved rather than enumerating, bounding the composed member count.
   */
  private static final int MAX_COMPOSITION_COMBINATIONS = 8;

  /**
   * Composes the output shape by parsing the equation and indexing each output label into the
   * corresponding input dimension. An input whose shape does not resolve contributes nothing to the
   * label bindings, and the axes only it names carry {@link UnresolvedDim} (wala/ML#737): the
   * equation still proves the output rank and every axis a resolved operand binds. Multi-shape
   * inputs enumerate as a capped cross-product, one composed member per combination. Returns ⊤
   * ({@code null}) when the equation is unresolved or malformed, when a combination is
   * unsatisfiable, or when the output's broadcast group depends on an unresolved operand.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The composed output shapes, or {@code null} (⊤) when they can't be resolved.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    String equation = this.getEquation(builder);
    if (equation == null) return null;

    ParsedEquation parsed = parseEquation(equation);
    if (parsed == null) return null;

    // The inputs are spread positionally after the equation (they only pack into the callee's
    // *inputs tuple), so input i is at positional index i + 1. A null entry marks an input whose
    // shape did not resolve.
    int inputCount = parsed.inputs().size();
    List<Set<List<Dimension<?>>>> inputShapeSets = new ArrayList<>(inputCount);
    long combinations = 1;
    for (int i = 0; i < inputCount; i++) {
      int position = Parameters.INPUTS.getIndex() + i;
      Set<List<Dimension<?>>> shapes = this.shapesOfArg(builder, position, null);
      if (shapes == null || shapes.isEmpty()) {
        inputShapeSets.add(null);
        continue;
      }
      inputShapeSets.add(shapes);
      combinations *= shapes.size();
    }

    // Over the cap, multi-shape inputs degrade to unresolved: the singleton inputs' evidence and
    // the equation's rank proof survive without enumerating.
    if (combinations > MAX_COMPOSITION_COMBINATIONS)
      for (int i = 0; i < inputCount; i++) {
        Set<List<Dimension<?>>> shapes = inputShapeSets.get(i);
        if (shapes != null && shapes.size() > 1) inputShapeSets.set(i, null);
      }

    Set<List<Dimension<?>>> outputs = HashSetFactory.make();
    List<List<Dimension<?>>> combination = new ArrayList<>(Collections.nCopies(inputCount, null));
    if (!composeCombinations(parsed, inputShapeSets, combination, 0, outputs)) return null;
    return outputs.isEmpty() ? null : outputs;
  }

  /**
   * Enumerates the cross-product of the inputs' shape possibilities, composing one output per
   * combination into {@code outputs}.
   *
   * @param parsed The parsed equation.
   * @param inputShapeSets Per-input shape possibilities; {@code null} for an unresolved input.
   * @param combination The combination under construction; index {@code i} holds input {@code i}'s
   *     pick, {@code null} for unresolved.
   * @param index The next input to pick for.
   * @param outputs The composed outputs, accumulated.
   * @return {@code false} iff some combination fails to compose, in which case the caller returns ⊤
   *     rather than asserting an incomplete member set.
   */
  private static boolean composeCombinations(
      ParsedEquation parsed,
      List<Set<List<Dimension<?>>>> inputShapeSets,
      List<List<Dimension<?>>> combination,
      int index,
      Set<List<Dimension<?>>> outputs) {
    if (index == inputShapeSets.size()) {
      List<Dimension<?>> output = composeOutputShape(parsed, combination);
      if (output == null) return false;
      outputs.add(output);
      return true;
    }
    Set<List<Dimension<?>>> shapes = inputShapeSets.get(index);
    if (shapes == null) {
      combination.set(index, null);
      return composeCombinations(parsed, inputShapeSets, combination, index + 1, outputs);
    }
    for (List<Dimension<?>> shape : shapes) {
      combination.set(index, shape);
      if (!composeCombinations(parsed, inputShapeSets, combination, index + 1, outputs))
        return false;
    }
    return true;
  }

  /**
   * Resolves the {@code equation} argument to a single constant string.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The equation string, or {@code null} if the argument isn't a single constant string.
   */
  private String getEquation(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(
            builder, Parameters.EQUATION.getIndex(), Parameters.EQUATION.getName());
    if (pts == null || pts.isEmpty()) return null;

    String found = null;
    for (InstanceKey ik : pts) {
      // Any non-string-constant possible value makes the equation unknown: fall back to ⊤ rather
      // than treating a mixed argument as a single constant.
      if (!(ik instanceof ConstantKey)) return null;
      Object value = ((ConstantKey<?>) ik).getValue();
      if (!(value instanceof String)) return null;
      if (found == null) found = (String) value;
      else if (!found.equals(value)) return null; // Ambiguous equation across contexts.
    }
    return found;
  }

  /** The parsed-label token representing a broadcasting ellipsis ({@code ...}) in a term. */
  private static final String ELLIPSIS = "...";

  /**
   * Parses an einsum equation into its per-input label lists and its output label list.
   *
   * <p>Supports explicit ({@code "ij,jk->ik"}) and implicit ({@code "ij,jk"}) output modes. In
   * implicit mode the output is the labels occurring exactly once, in alphabetical order (the
   * NumPy/TensorFlow convention), preceded by the broadcast group when any input term carries an
   * ellipsis. A broadcasting ellipsis parses to the {@link #ELLIPSIS} token, at most one per term.
   * Returns {@code null} for a malformed equation (a stray dot, a second ellipsis or arrow, or a
   * non-letter label).
   *
   * @param equation The raw equation string.
   * @return The parsed equation, or {@code null} if it is malformed.
   */
  private static ParsedEquation parseEquation(String equation) {
    String eq = equation.replace(" ", "");
    if (eq.isEmpty()) return null;

    String inputsPart;
    String outputPart;
    int arrow = eq.indexOf("->");
    if (arrow >= 0) {
      if (eq.indexOf("->", arrow + 2) >= 0) return null; // A second arrow is malformed.
      inputsPart = eq.substring(0, arrow);
      outputPart = eq.substring(arrow + 2);
    } else {
      inputsPart = eq;
      outputPart = null; // Implicit output mode.
    }

    List<List<String>> inputs = new ArrayList<>();
    for (String term : inputsPart.split(",", -1)) {
      List<String> labels = parseTerm(term);
      if (labels == null) return null;
      inputs.add(labels);
    }

    List<String> output;
    if (outputPart == null) {
      output = new ArrayList<>();
      // Implicit mode: the broadcast group first, then labels occurring exactly once,
      // alphabetical (TreeMap key order).
      if (inputs.stream().anyMatch(term -> term.contains(ELLIPSIS))) output.add(ELLIPSIS);
      Map<String, Integer> counts = new TreeMap<>();
      for (List<String> term : inputs)
        for (String label : term) if (!ELLIPSIS.equals(label)) counts.merge(label, 1, Integer::sum);
      for (Map.Entry<String, Integer> entry : counts.entrySet())
        if (entry.getValue() == 1) output.add(entry.getKey());
    } else {
      output = parseTerm(outputPart);
      if (output == null) return null;
      // TensorFlow/NumPy require output subscripts to be unique; a repeated output label (e.g.
      // "ij,jk->ii") is rejected at runtime, so don't infer a shape for it.
      for (int i = 0; i < output.size(); i++) if (output.indexOf(output.get(i)) != i) return null;
    }

    return new ParsedEquation(inputs, output);
  }

  /**
   * Tokenizes one equation term into its labels: single letters plus at most one {@link #ELLIPSIS}.
   *
   * @param term The term text, with spaces already stripped.
   * @return The label tokens, or {@code null} when the term is malformed (a stray dot, a second
   *     ellipsis, or a non-letter label).
   */
  private static List<String> parseTerm(String term) {
    List<String> labels = new ArrayList<>();
    for (int i = 0; i < term.length(); i++) {
      char c = term.charAt(i);
      if (c == '.') {
        if (labels.contains(ELLIPSIS)) return null; // A second ellipsis is malformed.
        if (i + 2 >= term.length() || term.charAt(i + 1) != '.' || term.charAt(i + 2) != '.')
          return null; // A dot outside an ellipsis is malformed.
        labels.add(ELLIPSIS);
        i += 2;
      } else if (isAsciiLetter(c)) labels.add(String.valueOf(c));
      else return null;
    }
    return labels;
  }

  /**
   * Composes the output shape by mapping each label to the input dimension it names, then indexing
   * the output labels into that map.
   *
   * <p>An input shape may encode a statically-unknown dimension size as a raw {@code null} entry
   * (see {@code TensorType.RaggedDim}'s Javadoc and wala/ML#414). Presence in the label map is
   * therefore tracked with {@code containsKey}, never by a null-check on the value: a label whose
   * occurrences are all {@code null} stays {@code null} (unknown size) in the output, preserving
   * the output's rank instead of degrading the whole shape to ⊤. A shared label names the same
   * dimension and the runtime requires the sizes equal, so a statically-known occurrence refines an
   * unknown or dynamic one (wala/ML#704); two known occurrences that disagree fall back to ⊤.
   *
   * <p>A repeated label within one term (diagonal/trace, wala/ML#705) names axes the runtime
   * requires equal, so its occurrences flow through the same refinement and contribute one output
   * dimension. Each term's ellipsis binds the axes its letters don't consume; the per-term groups
   * broadcast right-aligned, and the output's ellipsis receives the broadcast result. A nonempty
   * broadcast group with no ellipsis in the output is rejected at runtime, so no shape is inferred
   * for it.
   *
   * @param parsed The parsed equation.
   * @param inputShapes The resolved shape of each input, in input order; a {@code null} entry marks
   *     an unresolved input, which binds nothing (wala/ML#737).
   * @return The composed output shape (possibly containing {@code null} unknown-size entries), or
   *     {@code null} (⊤) when a term's label count doesn't match its input's rank, a shared label
   *     maps to conflicting known dimensions, the ellipsis groups don't broadcast, or an output
   *     label names no input.
   */
  private static List<Dimension<?>> composeOutputShape(
      ParsedEquation parsed, List<List<Dimension<?>>> inputShapes) {
    Map<String, Dimension<?>> labelToDim = new HashMap<>();
    List<Dimension<?>> broadcast = new ArrayList<>();
    boolean broadcastUnknown = false;

    for (int i = 0; i < parsed.inputs().size(); i++) {
      List<String> labels = parsed.inputs().get(i);
      List<Dimension<?>> shape = inputShapes.get(i);
      // An unresolved input binds nothing; if it carries the ellipsis, the broadcast group's
      // residual rank is unknowable through it (wala/ML#737).
      if (shape == null) {
        if (labels.contains(ELLIPSIS)) broadcastUnknown = true;
        continue;
      }
      int ellipsis = labels.indexOf(ELLIPSIS);
      int letters = ellipsis < 0 ? labels.size() : labels.size() - 1;
      int groupRank = shape.size() - letters;
      // Without an ellipsis the letters must consume the rank exactly; with one they may not
      // exceed it.
      if (ellipsis < 0 ? groupRank != 0 : groupRank < 0) return null;

      for (int d = 0; d < labels.size(); d++) {
        String label = labels.get(d);
        if (ELLIPSIS.equals(label)) continue;
        // Letters before the ellipsis bind leading axes; letters after it bind trailing ones.
        Dimension<?> dim = shape.get(ellipsis >= 0 && d > ellipsis ? d - 1 + groupRank : d);
        if (!bindLabelOccurrence(labelToDim, label, dim)) return null;
      }

      if (ellipsis >= 0) {
        broadcast = mergeBroadcastGroups(broadcast, shape.subList(ellipsis, ellipsis + groupRank));
        if (broadcast == null) return null; // Known sizes that don't broadcast.
      }
    }

    List<Dimension<?>> output = new ArrayList<>(parsed.output().size());
    for (String label : parsed.output()) {
      if (ELLIPSIS.equals(label)) {
        // The group's rank could exceed the resolved inputs' merge through an unresolved
        // operand's residual axes, so the output rank is unknowable (wala/ML#737).
        if (broadcastUnknown) return null;
        output.addAll(broadcast);
        continue;
      }
      if (labelToDim.containsKey(label)) {
        output.add(labelToDim.get(label)); // May be null: known rank, unknown size.
        continue;
      }
      // The label is named only by unresolved operands: the axis exists with a fixed runtime size
      // the analysis could not compute, with no None evidence (wala/ML#721, wala/ML#737).
      if (parsed.inputs().stream().anyMatch(term -> term.contains(label)))
        output.add(UnresolvedDim.INSTANCE);
      else return null; // Output label names no input.
    }
    // Broadcast axes with no ellipsis to receive them make the equation unsatisfiable (the
    // runtime rejects it), so no shape is inferred.
    if (!broadcast.isEmpty() && !parsed.output().contains(ELLIPSIS)) return null;
    return output;
  }

  /**
   * Binds one occurrence of a subscript label to the dimension it names. A shared label names the
   * same dimension, and einsum requires the sizes equal at runtime, so a statically-known
   * occurrence refines an unknown or dynamic one (wala/ML#704: the contracted hidden dim is known
   * on the input but dynamic on the reshaped weight). Two known occurrences that disagree mean the
   * constraint isn't satisfiable.
   *
   * @param labelToDim The bindings accumulated so far; updated in place.
   * @param label The subscript label.
   * @param dim The dimension this occurrence names; may be {@code null} (known rank, unknown size).
   * @return {@code false} iff this occurrence conflicts with a known binding (two unequal known
   *     sizes).
   */
  private static boolean bindLabelOccurrence(
      Map<String, Dimension<?>> labelToDim, String label, Dimension<?> dim) {
    if (!labelToDim.containsKey(label)) {
      labelToDim.put(label, dim);
      return true;
    }
    Dimension<?> previous = labelToDim.get(label);
    boolean previousKnown = previous instanceof NumericDim;
    boolean dimKnown = dim instanceof NumericDim;
    if (previousKnown && dimKnown) return previous.equals(dim);
    if (dimKnown) labelToDim.put(label, dim);
    else if (!previousKnown && previous == null && dim != null) labelToDim.put(label, dim);
    return true;
  }

  /**
   * Derives, for each tensor operand whose own shape does not resolve, the shape the equation
   * proves for it: the operand's term fixes its rank (when the term carries no ellipsis), and each
   * label shared with an operand whose shape is statically known fixes that axis's extent, since
   * einsum requires same-label sizes equal at runtime (wala/ML#704). An axis the equation leaves
   * unconstrained carries {@link UnresolvedDim}: its size is a fixed runtime value the analysis
   * could not compute, with no {@code None} evidence (wala/ML#721); an axis bound to another
   * operand's {@link DynamicDim} or {@link TensorType.RaggedDim} keeps that evidence.
   *
   * <p>The einsum source is the synthetic summary's return value, so the operands live caller-side:
   * each caller's invoke supplies its own operand value numbers, resolved in that caller's frame
   * (the {@code getArgumentShapesViaCallers} walk). Operand shapes are read in exact mode
   * (wala/ML#716): a label binding asserts an equality on another operand, so a partially resolved
   * shape set is not proof.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return A map from each caller-side operand's {@link PointerKey} to the shape the equation
   *     proves for it; empty when the equation does not resolve, every operand resolves on its own,
   *     or a resolved operand contradicts the equation. An operand whose call sites prove
   *     disagreeing constraints is omitted.
   */
  public Map<PointerKey, List<Dimension<?>>> getOperandShapeConstraints(
      PropagationCallGraphBuilder builder) {
    String equation = this.getEquation(builder);
    LOGGER.fine(() -> "getOperandShapeConstraints: equation=" + equation);
    if (equation == null) return Collections.emptyMap();

    ParsedEquation parsed = parseEquation(equation);
    if (parsed == null) return Collections.emptyMap();
    int inputCount = parsed.inputs().size();

    Map<PointerKey, List<Dimension<?>>> ret = new LinkedHashMap<>();
    Set<PointerKey> conflicting = new HashSet<>();
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      CGNode caller = callerInvoke.fst;
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction invoke = (PythonInvokeInstruction) callerInvoke.snd;
      // The *inputs varargs are spread positionally after the equation (input i sits at
      // positional index i + 1, i.e., use i + 2); a call site with fewer positional arguments
      // than the equation's terms is left alone.
      if (invoke.getNumberOfPositionalParameters() < inputCount + 2) continue;

      List<Set<List<Dimension<?>>>> operandShapes = new ArrayList<>(inputCount);
      for (int i = 0; i < inputCount; i++) {
        int argVn = invoke.getUse(Parameters.INPUTS.getIndex() + i + 1);
        Set<List<Dimension<?>>> shapes = this.getShapes(builder, caller, argVn, true);
        final int index = i;
        LOGGER.fine(() -> "getOperandShapeConstraints: input " + index + " shapes=" + shapes);
        operandShapes.add(shapes);
      }

      // Bind the labels of the operands whose own shape resolves to a single known form. A
      // resolved operand whose rank contradicts its term means the call fails at runtime; so
      // does a shared label with two unequal known sizes. No constraint is proven either way.
      Map<String, Dimension<?>> labelToDim = new HashMap<>();
      boolean contradicted = false;
      for (int i = 0; i < inputCount && !contradicted; i++) {
        Set<List<Dimension<?>>> shapes = operandShapes.get(i);
        if (shapes == null || shapes.size() != 1) continue;
        List<String> labels = parsed.inputs().get(i);
        if (labels.contains(ELLIPSIS)) continue;
        List<Dimension<?>> shape = shapes.iterator().next();
        if (shape.size() != labels.size()) contradicted = true;
        for (int d = 0; d < labels.size() && !contradicted; d++)
          contradicted = !bindLabelOccurrence(labelToDim, labels.get(d), shape.get(d));
      }
      if (contradicted) continue;

      for (int i = 0; i < inputCount; i++) {
        Set<List<Dimension<?>>> shapes = operandShapes.get(i);
        if (shapes != null && shapes.size() == 1)
          continue; // Resolves on its own; nothing to prove.
        List<String> labels = parsed.inputs().get(i);
        if (labels.contains(ELLIPSIS)) continue; // The term does not fix the operand's rank.
        List<Dimension<?>> constraint = new ArrayList<>(labels.size());
        for (String label : labels) {
          Dimension<?> dim = labelToDim.get(label);
          constraint.add(
              dim instanceof NumericDim || dim instanceof DynamicDim || dim instanceof RaggedDim
                  ? dim
                  : UnresolvedDim.INSTANCE);
        }
        PointerKey operandKey =
            builder
                .getPointerAnalysis()
                .getHeapModel()
                .getPointerKeyForLocal(caller, invoke.getUse(Parameters.INPUTS.getIndex() + i + 1));
        List<Dimension<?>> previous = ret.putIfAbsent(operandKey, constraint);
        if (previous != null && !previous.equals(constraint)) conflicting.add(operandKey);
      }
    }
    ret.keySet().removeAll(conflicting);
    return ret;
  }

  /**
   * Broadcasts two ellipsis dimension groups right-aligned, per the NumPy/TensorFlow rules: absent
   * axes act as size 1, a known size-1 axis yields the other side, and equal sizes yield
   * themselves. A known non-1 size against a statically-unknown axis yields the known size (the
   * runtime requires the unknown to be 1 or equal); two unequal known non-1 sizes don't broadcast.
   * Statically-unknown pairs yield a raw {@code null} unknown-size entry unless equal.
   *
   * @param a The accumulated broadcast group.
   * @param b The next term's ellipsis group.
   * @return The broadcast group, or {@code null} when two known sizes don't broadcast.
   */
  private static List<Dimension<?>> mergeBroadcastGroups(
      List<Dimension<?>> a, List<Dimension<?>> b) {
    int rank = Math.max(a.size(), b.size());
    List<Dimension<?>> result = new ArrayList<>(Collections.nCopies(rank, null));
    for (int i = 0; i < rank; i++) {
      boolean inA = i < a.size();
      boolean inB = i < b.size();
      Dimension<?> da = inA ? a.get(a.size() - 1 - i) : null;
      Dimension<?> db = inB ? b.get(b.size() - 1 - i) : null;
      Dimension<?> merged;
      if (!inA) merged = db; // Absent axes broadcast as size 1.
      else if (!inB) merged = da;
      else if (da instanceof NumericDim && db instanceof NumericDim) {
        if (da.equals(db)) merged = da;
        else if (Integer.valueOf(1).equals(da.value())) merged = db;
        else if (Integer.valueOf(1).equals(db.value())) merged = da;
        else return null; // Unequal known non-1 sizes don't broadcast.
      } else if (da instanceof NumericDim) merged = Integer.valueOf(1).equals(da.value()) ? db : da;
      else if (db instanceof NumericDim) merged = Integer.valueOf(1).equals(db.value()) ? da : db;
      else merged = Objects.equals(da, db) ? da : null; // Unknown pair: unknown size.
      result.set(rank - 1 - i, merged);
    }
    return result;
  }

  /**
   * Returns whether the character is an ASCII letter. Einsum subscript labels are restricted to
   * {@code [A-Za-z]} by TensorFlow/NumPy (aside from the ellipsis), so a Unicode letter accepted by
   * {@link Character#isLetter(char)} would type an equation the runtime rejects.
   *
   * @param c The candidate label character.
   * @return {@code true} iff {@code c} is in {@code [A-Za-z]}.
   */
  private static boolean isAsciiLetter(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  }

  /**
   * A parsed einsum equation: the label list of each input term and the output label list.
   *
   * @param inputs The per-input-term label lists, in input order.
   * @param output The output label list.
   */
  private record ParsedEquation(List<List<String>> inputs, List<String> output) {}

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
   * This generator transforms its input's shape, so forwarding operand shapes would overclaim; the
   * feed carries dtype only (wala/ML#682).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The dtype-only feed over the caller-side input keys, or {@code null} when none is
   *     located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return this.getTypeFeed(builder, TypeFeedKind.DTYPE_ONLY);
  }
}
