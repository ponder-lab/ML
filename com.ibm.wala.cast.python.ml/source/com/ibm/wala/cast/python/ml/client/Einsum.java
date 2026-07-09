package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Generator for {@code tf.einsum(equation, *inputs, **kwargs)}. Parses the equation string (e.g.
 * {@code "ij,jk->ik"}) and composes the precise output shape from each input's shape: every output
 * label is resolved to the input dimension it names. Output dtype inherits from the first tensor
 * input (TF promotes per-input dtypes upstream of einsum, so the first input's dtype is the
 * canonical source).
 *
 * <p>Falls back to ⊤ (unknown shape) when the equation can't be resolved to a single constant
 * string, uses broadcasting ellipsis ({@code ...}) or a repeated (diagonal) label within one term,
 * or when any input's shape is itself ⊤ — the sound answer in those cases.
 *
 * <p>Argument layout at the call site: the {@code *inputs} varargs are spread positionally after
 * the equation (the packing into a single {@code inputs} tuple is callee-side only), so:
 *
 * <ul>
 *   <li>position 0: {@code equation} (string).
 *   <li>position 1: the first tensor input; the dtype source.
 *   <li>position 2, 3, ...: the remaining tensor inputs, one per equation term.
 * </ul>
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/linalg/einsum">tf.linalg.einsum</a>
 * @see <a href="https://github.com/wala/ML/issues/507">wala/ML#507</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Einsum extends PassThroughUnaryTensorGenerator {

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
   * Composes the precise output shape by parsing the equation and indexing each output label into
   * the corresponding input dimension. Returns ⊤ (unknown shape, {@code null}) when the equation or
   * any input shape can't be resolved statically, or when the equation uses a form this parser
   * doesn't model (ellipsis, diagonal labels).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The single composed output shape, or {@code null} (⊤) when it can't be resolved.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    String equation = this.getEquation(builder);
    if (equation == null) return null;

    ParsedEquation parsed = parseEquation(equation);
    if (parsed == null) return null;

    // The inputs are spread positionally after the equation (they only pack into the callee's
    // *inputs tuple), so input i is at positional index i + 1.
    int inputCount = parsed.inputs().size();
    List<List<Dimension<?>>> inputShapes = new ArrayList<>(inputCount);
    for (int i = 0; i < inputCount; i++) {
      int position = Parameters.INPUTS.getIndex() + i;
      Set<List<Dimension<?>>> shapes = this.shapesOfArg(builder, position, null);
      // Require a single known shape per input. A ⊤ or multi-shape input yields a ⊤ output; the
      // multi-shape cross-product is deferred as a precision follow-up.
      if (shapes == null || shapes.size() != 1) return null;
      inputShapes.add(shapes.iterator().next());
    }

    List<Dimension<?>> outputShape = composeOutputShape(parsed, inputShapes);
    return outputShape == null ? null : Collections.singleton(outputShape);
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
      if (!(ik instanceof ConstantKey)) continue;
      Object value = ((ConstantKey<?>) ik).getValue();
      if (!(value instanceof String)) continue;
      if (found == null) found = (String) value;
      else if (!found.equals(value)) return null; // Ambiguous equation across contexts.
    }
    return found;
  }

  /**
   * Parses an einsum equation into its per-input label lists and its output label list.
   *
   * <p>Supports explicit ({@code "ij,jk->ik"}) and implicit ({@code "ij,jk"}) output modes. In
   * implicit mode the output is the labels occurring exactly once, in alphabetical order (the
   * NumPy/TensorFlow convention). Returns {@code null} for forms this parser doesn't model:
   * broadcasting ellipsis ({@code ...}), a malformed equation, or a non-letter label.
   *
   * @param equation The raw equation string.
   * @return The parsed equation, or {@code null} if it uses an unsupported or malformed form.
   */
  private static ParsedEquation parseEquation(String equation) {
    String eq = equation.replace(" ", "");
    if (eq.isEmpty() || eq.contains("...")) return null; // Broadcasting ellipsis: ⊤.

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
      List<String> labels = new ArrayList<>();
      for (int i = 0; i < term.length(); i++) {
        char c = term.charAt(i);
        if (!Character.isLetter(c)) return null;
        labels.add(String.valueOf(c));
      }
      inputs.add(labels);
    }

    List<String> output = new ArrayList<>();
    if (outputPart == null) {
      // Implicit mode: labels occurring exactly once, alphabetical (TreeMap key order).
      Map<String, Integer> counts = new TreeMap<>();
      for (List<String> term : inputs)
        for (String label : term) counts.merge(label, 1, Integer::sum);
      for (Map.Entry<String, Integer> entry : counts.entrySet())
        if (entry.getValue() == 1) output.add(entry.getKey());
    } else {
      for (int i = 0; i < outputPart.length(); i++) {
        char c = outputPart.charAt(i);
        if (!Character.isLetter(c)) return null;
        output.add(String.valueOf(c));
      }
    }

    return new ParsedEquation(inputs, output);
  }

  /**
   * Composes the output shape by mapping each label to the input dimension it names, then indexing
   * the output labels into that map.
   *
   * @param parsed The parsed equation.
   * @param inputShapes The resolved shape of each input, in input order.
   * @return The composed output shape, or {@code null} when a term's label count doesn't match its
   *     input's rank, a term repeats a label (diagonal, unmodeled), or an output label names no
   *     input.
   */
  private static List<Dimension<?>> composeOutputShape(
      ParsedEquation parsed, List<List<Dimension<?>>> inputShapes) {
    Map<String, Dimension<?>> labelToDim = new HashMap<>();

    for (int i = 0; i < parsed.inputs().size(); i++) {
      List<String> labels = parsed.inputs().get(i);
      List<Dimension<?>> shape = inputShapes.get(i);
      if (labels.size() != shape.size()) return null; // Rank mismatch: labels vs. shape.

      Set<String> seenInTerm = new HashSet<>();
      for (int d = 0; d < labels.size(); d++) {
        String label = labels.get(d);
        if (!seenInTerm.add(label)) return null; // Repeated label within a term (diagonal): ⊤.
        labelToDim.putIfAbsent(label, shape.get(d));
      }
    }

    List<Dimension<?>> output = new ArrayList<>(parsed.output().size());
    for (String label : parsed.output()) {
      Dimension<?> dim = labelToDim.get(label);
      if (dim == null) return null; // Output label names no input.
      output.add(dim);
    }
    return output;
  }

  /**
   * A parsed einsum equation: the label list of each input term and the output label list.
   *
   * @param inputs The per-input-term label lists, in input order.
   * @param output The output label list.
   */
  private record ParsedEquation(List<List<String>> inputs, List<String> output) {}
}
