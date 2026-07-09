package com.ibm.wala.cast.python.ml.test;

import com.ibm.wala.cast.python.ml.util.ShortClassNameFormatter;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Formatter;
import java.util.logging.Handler;
import java.util.logging.LogRecord;

/**
 * Formats every published record (exercising the full diagnostic-logging render path) but discards
 * the output, tracking only the cumulative formatted volume.
 *
 * <p>Used by the wala/ML#702 diagnostic-logging volume guard. Correct code keeps every render
 * bounded via {@code Loggables.describe(...)}, so FINEST volume stays small; a bare render of a
 * {@code Context}-bearing value inflates a message by orders of magnitude, which the guard detects
 * as excess volume — reproducing the wala/ML#697 failure that is invisible at CI's {@code WARNING}
 * level, without depending on an actual {@code OutOfMemoryError} (whose timing varies with heap and
 * output backpressure).
 */
public final class DiscardingFormattingHandler extends Handler {

  private static final AtomicLong TOTAL_CHARS = new AtomicLong();

  private final Formatter formatter = new ShortClassNameFormatter();

  /** Returns the cumulative formatted-character count across all instances. */
  public static long totalChars() {
    return TOTAL_CHARS.get();
  }

  /** Resets the cumulative count; call before a measured run. */
  public static void reset() {
    TOTAL_CHARS.set(0L);
  }

  @Override
  public void publish(LogRecord record) {
    if (!isLoggable(record)) return;
    // Format (not merely build the message) so the guard covers the full publish-path render, then
    // discard the result — only its length feeds the running total.
    TOTAL_CHARS.addAndGet(formatter.format(record).length());
  }

  @Override
  public void flush() {}

  @Override
  public void close() {}
}
