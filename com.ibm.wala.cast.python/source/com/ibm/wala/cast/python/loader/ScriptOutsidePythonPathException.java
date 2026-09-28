package com.ibm.wala.cast.python.loader;

import java.io.File;
import java.util.Collections;
import java.util.List;

/**
 * A script the analysis was given lies outside every PYTHONPATH entry, so it cannot be bound to a
 * module (wala/ML#977). The source path is the project's configuration, so the client decides how
 * to deal with it: the exception carries the script and the path, and it propagates out of the
 * class-hierarchy build as itself.
 */
public class ScriptOutsidePythonPathException extends IllegalStateException {

  private static final long serialVersionUID = 1L;

  private final String script;

  /** Transient because a {@link List} is not serializable; the message keeps the path's text. */
  private final transient List<File> pythonPath;

  /**
   * Creates the exception for the given script and path.
   *
   * @param script The name of the script (its module entry's name), as the analysis sees it.
   * @param pythonPath The PYTHONPATH in effect, none of whose entries contains the script.
   * @param detail A sentence saying what the analysis needed the script's module for, or {@code
   *     null}.
   */
  public ScriptOutsidePythonPathException(String script, List<File> pythonPath, String detail) {
    super(
        "Cannot find script: "
            + script
            + " in PYTHONPATH: "
            + pythonPath
            + (detail == null ? "." : "; " + detail));
    this.script = script;
    this.pythonPath =
        pythonPath == null ? Collections.emptyList() : Collections.unmodifiableList(pythonPath);
  }

  /**
   * The name of the script that no PYTHONPATH entry contains.
   *
   * @return The script's name, as the analysis sees it.
   */
  public String getScript() {
    return script;
  }

  /**
   * The PYTHONPATH in effect when the script was translated.
   *
   * @return The path entries; empty when none were configured.
   */
  public List<File> getPythonPath() {
    return pythonPath;
  }
}
