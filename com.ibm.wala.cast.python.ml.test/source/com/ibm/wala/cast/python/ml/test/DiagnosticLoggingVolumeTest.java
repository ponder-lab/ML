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
   * <p>Correct code renders diagnostics through the bounded {@code Loggables.describe(...)} (~0.6
   * GB of formatted {@code FINEST} output for this analysis); a bare render of a {@code
   * Context}-bearing value inflates that by more than an order of magnitude (~14 GB measured),
   * which this bound catches. The failure is invisible at CI's {@code WARNING} level (the message
   * strings are never built), so this test is the pipeline's guard against its return. See {@code
   * CONTRIBUTING.md}'s Diagnostic Logging section and <a
   * href="https://github.com/wala/WALA/issues/1992">wala/WALA#1992</a> for the upstream root cause.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testDiagnosticLoggingVolumeBounded()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // Fixed code emits ~0.6 GB of formatted FINEST volume for this analysis; a bare `Context`
    // render measured ~14 GB. 2 GB sits well above the former and far below the latter.
    final long maxFormattedChars = 2_000_000_000L;

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
