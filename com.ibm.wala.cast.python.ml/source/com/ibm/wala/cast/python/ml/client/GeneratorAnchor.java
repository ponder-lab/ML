package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
import java.util.Set;
import java.util.function.Function;

/**
 * An anchoring for a {@link TensorGenerator} under construction (wala/ML#760): either
 * <em>source-based</em> (a {@link PointsToSetVariable} whose defining call site names the
 * operation, dispatched by {@link TensorGeneratorFactory#getGenerator}) or <em>manual</em> (the
 * operation's allocating synthetic {@link CGNode}, dispatched by {@code
 * TensorGenerator.createManualGenerator} during producer delegation and receiver walks).
 *
 * <p>The shared dispatch chain ({@code TensorGeneratorFactory.dispatchShared}) matches an arm on
 * {@link #declaredType()} and constructs through {@link #makeGenerator}, so every migrated arm
 * names both construction forms in one place: an operation registered for one anchoring cannot
 * silently lack the other, the failure class behind the wala/ML#757 and wala/ML#759 delegation
 * dead-ends. An operation that genuinely has no form for one anchoring declares that in its arm
 * rather than by omission.
 */
public sealed interface GeneratorAnchor {

  /**
   * The declared type the dispatch matches on: the (sanitized) called function's type for a source
   * anchoring, the allocating node's declaring class for a manual one.
   *
   * @return The declared type.
   */
  TypeReference declaredType();

  /**
   * The propagation call graph builder in scope for this construction.
   *
   * @return The builder.
   */
  PropagationCallGraphBuilder builder();

  /**
   * Constructs the generator through the form matching this anchoring.
   *
   * @param fromSource The source-anchored construction.
   * @param fromNode The node-anchored construction.
   * @return The constructed generator.
   */
  TensorGenerator makeGenerator(
      Function<PointsToSetVariable, TensorGenerator> fromSource,
      Function<CGNode, TensorGenerator> fromNode);

  /**
   * A source-based anchoring (wala/ML#760).
   *
   * @param source The {@link PointsToSetVariable} whose defining call produces the value.
   * @param declaredType The sanitized called function's type.
   * @param builder The propagation call graph builder.
   * @param visited The factory's creator-walk visited set, for arms whose source path consults
   *     operand evidence (e.g. the wala/ML#451 binary-operation gate).
   */
  record SourceAnchor(
      PointsToSetVariable source,
      TypeReference declaredType,
      PropagationCallGraphBuilder builder,
      Set<PointsToSetVariable> visited)
      implements GeneratorAnchor {
    @Override
    public TensorGenerator makeGenerator(
        Function<PointsToSetVariable, TensorGenerator> fromSource,
        Function<CGNode, TensorGenerator> fromNode) {
      return fromSource.apply(this.source());
    }
  }

  /**
   * A manual (node-based) anchoring (wala/ML#760).
   *
   * @param node The allocating synthetic {@code do()} {@link CGNode}.
   * @param declaredType The node's declaring class, sanitized of the trampoline suffix.
   * @param builder The propagation call graph builder.
   */
  record NodeAnchor(CGNode node, TypeReference declaredType, PropagationCallGraphBuilder builder)
      implements GeneratorAnchor {
    @Override
    public TensorGenerator makeGenerator(
        Function<PointsToSetVariable, TensorGenerator> fromSource,
        Function<CGNode, TensorGenerator> fromNode) {
      return fromNode.apply(this.node());
    }
  }
}
