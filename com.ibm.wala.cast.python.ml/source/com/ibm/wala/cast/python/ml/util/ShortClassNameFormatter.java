package com.ibm.wala.cast.python.ml.util;

import java.util.logging.Formatter;
import java.util.logging.LogRecord;

/**
 * A {@link Formatter} that emits log records with the simple (unqualified) class name of the
 * emitting logger, preserving per-class provenance in the output without the visual width of the
 * fully qualified name.
 *
 * <p>Output format per record:
 *
 * <pre>
 *   [LEVEL] [SimpleClassName] message
 * </pre>
 *
 * <p>Example: a record emitted by a logger named {@code com.ibm.wala.cast.python.ml.client.Ones}
 * renders as {@code [FINE] [Ones] ...}.
 *
 * <p>The underlying logger name remains the full FQN in the {@link LogRecord}, so package-hierarchy
 * level filtering via {@code logging.properties} (e.g., {@code
 * com.ibm.wala.cast.python.ml.client.level=FINER}) still works — this class only changes what the
 * formatter prints, not how the logging framework routes records.
 *
 * <p>Wired in via {@code logging.properties} / {@code logging.ci.properties}:
 *
 * <pre>
 *   java.util.logging.ConsoleHandler.formatter =
 *       com.ibm.wala.cast.python.ml.util.ShortClassNameFormatter
 * </pre>
 */
public class ShortClassNameFormatter extends Formatter {

  @Override
  public String format(LogRecord record) {
    String loggerName = record.getLoggerName();
    String shortName = loggerName == null ? "?" : simpleName(loggerName);
    return "["
        + record.getLevel().getName()
        + "] ["
        + shortName
        + "] "
        + formatMessage(record)
        + System.lineSeparator();
  }

  /** Returns everything after the last {@code '.'} in {@code fqn}, or {@code fqn} if no dot. */
  private static String simpleName(String fqn) {
    int dot = fqn.lastIndexOf('.');
    return dot < 0 ? fqn : fqn.substring(dot + 1);
  }
}
