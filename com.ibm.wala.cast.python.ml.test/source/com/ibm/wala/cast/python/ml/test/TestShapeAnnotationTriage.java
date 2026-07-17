package com.ibm.wala.cast.python.ml.test;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine.ShapeAnnotationCandidate;
import com.ibm.wala.cast.python.ml.client.TensorGenerator.ShapeUnresolutionCause;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.Test;

/**
 * Regression guards for the wala/ML#370 shape-annotation triage (<a
 * href="https://github.com/wala/ML/issues/735">wala/ML#735</a>): {@link
 * PythonTensorAnalysisEngine#getShapeAnnotationCandidates()} classifies each allocator whose shape
 * argument did not resolve as content-dependent (a genuine annotation candidate) or a recoverable
 * precision gap, so the "candidate for a wala/ML#370 shape annotation" suggestion no longer fires
 * for statically-recoverable shapes.
 */
public class TestShapeAnnotationTriage extends TestPythonMLCallGraphShape {

  private Map<ShapeUnresolutionCause, Long> triage(String file)
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonTensorAnalysisEngine engine = makeEngine(emptyList(), file);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    List<ShapeAnnotationCandidate> candidates = engine.getShapeAnnotationCandidates();
    return candidates.stream()
        .collect(Collectors.groupingBy(ShapeAnnotationCandidate::cause, Collectors.counting()));
  }

  /**
   * The content-dependent allocator ({@code tf.zeros(np.load(...).shape)}) is a wala/ML#370
   * annotation candidate, while the modeled-op allocator ({@code
   * tf.zeros(tf.transpose(...).shape)}) is a recoverable precision gap rather than an annotation
   * candidate.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testShapeAnnotationTriage()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Map<ShapeUnresolutionCause, Long> byCause = triage("tf2_test_shape_annotation_triage.py");
    // The tf.zeros(np.load(...).shape) allocator is a genuine content-dependent wala/ML#370
    // candidate; the tf.zeros(tf.transpose(...).shape) allocator, whose shape roots in a modeled
    // op, is a recoverable precision gap and not a candidate.
    assertEquals(
        "tf.zeros(np.load(...).shape) is a content-dependent wala/ML#370 candidate.",
        1L,
        (long) byCause.getOrDefault(ShapeUnresolutionCause.CONTENT_DEPENDENT, 0L));
    assertEquals(
        "tf.zeros(tf.transpose(...).shape) is a recoverable precision gap, not a candidate.",
        1L,
        (long) byCause.getOrDefault(ShapeUnresolutionCause.RECOVERABLE_GAP, 0L));
  }
}
