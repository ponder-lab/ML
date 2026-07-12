package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertTrue;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.Test;

/**
 * Diagnostic-logging volume guard (<a
 * href="https://github.com/wala/ML/issues/702">wala/ML#702</a>): an infrastructure test, separate
 * from the tensor-model assertions in {@link TestTensorflow2Model}, that bounds the formatted
 * {@code FINEST} volume of a whole-project analysis.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DiagnosticLoggingVolumeTest {

  /**
   * Reruns the nlpgnn generation analysis — the cyclic-closure call graph that triggered <a
   * href="https://github.com/wala/ML/issues/697">wala/ML#697</a> — with every {@code
   * com.ibm.wala.cast.python} logger at {@code FINEST}, routing records through {@link
   * DiscardingFormattingHandler}, which formats and discards them while summing their volume.
   *
   * <p>Correct code renders diagnostics through the bounded {@code Loggables.describe(...)} (~2.2
   * GB of formatted {@code FINEST} output for this analysis as of wala/ML#714; ~0.6 GB at the
   * wala/ML#697 calibration); a bare render of a {@code Context}-bearing value inflates that by an
   * order of magnitude (~14 GB measured), which this bound catches. The failure is invisible at
   * CI's {@code WARNING} level (the message strings are never built), so this test is the
   * pipeline's guard against its return. See {@code CONTRIBUTING.md}'s Diagnostic Logging section,
   * <a href="https://github.com/wala/WALA/issues/1992">wala/WALA#1992</a> for the upstream root
   * cause, and <a href="https://github.com/wala/ML/issues/715">wala/ML#715</a> for the
   * re-baselining.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDiagnosticLoggingVolumeBounded()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // Fixed code emitted ~0.6 GB of formatted FINEST volume for this analysis when the guard was
    // calibrated (wala/ML#697, July 8); the shape-provenance machinery merged since (wala/ML#703
    // through wala/ML#714) organically grew that to ~2.2 GB measured in-suite on a clean master,
    // with run-to-run variance around the old 2 GB bound (wala/ML#715), and the tf.reshape /
    // tf.squeeze producer registrations plus the callee-return descent (wala/ML#718) grew it again
    // to ~5.6 GB — the same messages over roughly 2.5x the delegation traversals, no new render
    // sites. A bare `Context` render, the failure mode this guard exists to catch, measured ~14 GB
    // before that growth and scales with the same traversal count, so 8 GB keeps the gap.
    final long maxFormattedChars = 8_000_000_000L;

    Logger pkg = Logger.getLogger("com.ibm.wala.cast.python");
    DiscardingFormattingHandler handler = new DiscardingFormattingHandler();
    handler.setLevel(Level.ALL);
    Level oldLevel = pkg.getLevel();
    boolean oldUseParentHandlers = pkg.getUseParentHandlers();
    pkg.addHandler(handler);
    pkg.setLevel(Level.FINEST);
    // Count the FINEST volume here; don't also propagate it to the console handlers.
    pkg.setUseParentHandlers(false);
    try {
      DiscardingFormattingHandler.reset();
      new TestTensorflow2Model().runNlpgnnFullGeneration();
      long volume = DiscardingFormattingHandler.totalChars();
      assertTrue(
          "Diagnostic FINEST volume "
              + volume
              + " chars exceeds the "
              + maxFormattedChars
              + "-char bound; a pointer-analysis or call-graph value is likely logged without"
              + " Loggables.describe(...). See CONTRIBUTING.md's Diagnostic Logging section.",
          volume < maxFormattedChars);
    } finally {
      pkg.removeHandler(handler);
      pkg.setLevel(oldLevel);
      pkg.setUseParentHandlers(oldUseParentHandlers);
    }
  }
}
