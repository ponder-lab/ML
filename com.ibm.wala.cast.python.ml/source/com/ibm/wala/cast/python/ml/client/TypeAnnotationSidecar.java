package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.logging.Logger;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * Loader for per-project type-annotation sidecar files (wala/ML#370): user-supplied shape/dtype
 * facts for tensor values whose types are content-dependent (opaque reads such as {@code
 * pickle.load} or {@code np.load}) and therefore unreachable by static inference. The sidecar
 * leaves the analyzed program untouched (no imports, no restructuring, no runtime dependency), can
 * address any named binding including tuple-unpack targets, and is toggleable per run.
 *
 * <p>A sidecar is a JSON file named {@value #SIDECAR_FILE_NAME} at a Python-path root (or the file
 * named by the {@value #SIDECAR_FILE_PROPERTY} system property), of the form:
 *
 * <pre>{@code
 * { "types": [ { "module": "datas/loader.py", "function": "Planetoid.load",
 *                "variable": "allx", "dtype": "float32", "shape": "1708 3703" } ] }
 * }</pre>
 *
 * <p>{@code module} is the root-relative script path; {@code function} is the dotted qualified name
 * within the module, empty for module level; {@code variable} is the source-level name of the
 * annotated binding. {@code dtype} and {@code shape} are each optional (an absent axis stays
 * unknown). The shape vocabulary is the dimension string: whitespace-separated integers ({@code
 * NumericDim}) or names ({@code SymbolicDim}); {@code "..."} alone declares an unknown rank; the
 * empty string declares rank 0.
 *
 * <p>Consumption semantics live in {@link PythonTensorAnalysisEngine}: fill-only (an entry seeds
 * only where inference has no type), check-and-report on conflicts, and a dedicated {@link
 * com.ibm.wala.cast.python.ml.types.TensorOrigin#ANNOTATION} provenance, so annotations can extend
 * the analysis but never silently change or mask what it computes.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TypeAnnotationSidecar {

  private static final Logger LOGGER = Logger.getLogger(TypeAnnotationSidecar.class.getName());

  /** The sidecar file name searched for at each Python-path root. */
  public static final String SIDECAR_FILE_NAME = "ariadne-types.json";

  /** System property naming an explicit sidecar file, overriding the path search. */
  public static final String SIDECAR_FILE_PROPERTY = "ariadne.typeAnnotations.file";

  /**
   * One sidecar entry: a user-supplied type fact for a named binding.
   *
   * @param module The root-relative script path (e.g. {@code datas/loader.py}).
   * @param function The dotted qualified function name within the module; empty for module level.
   * @param variable The source-level name of the annotated binding.
   * @param type The declared tensor type (unknown axes per the entry's omissions).
   */
  public record Entry(String module, String function, String variable, TensorType type) {

    /**
     * Renders the entry's program-point anchor for logs and reports.
     *
     * @return The {@code module:function:variable} anchor.
     */
    public String anchor() {
      return this.module() + ":" + this.function() + ":" + this.variable();
    }
  }

  private TypeAnnotationSidecar() {}

  /**
   * Loads every sidecar entry visible to an analysis: the file named by {@value
   * #SIDECAR_FILE_PROPERTY} when set, else {@value #SIDECAR_FILE_NAME} at each given Python-path
   * root. A malformed file or entry is reported and skipped rather than aborting the analysis.
   *
   * @param pythonPath The Python-path roots to search.
   * @return The parsed entries, empty when no sidecar exists.
   */
  public static List<Entry> load(List<File> pythonPath) {
    List<File> candidates = new ArrayList<>();
    String explicit = System.getProperty(SIDECAR_FILE_PROPERTY);
    if (explicit != null) candidates.add(new File(explicit));
    else if (pythonPath != null)
      for (File root : pythonPath) candidates.add(new File(root, SIDECAR_FILE_NAME));

    List<Entry> ret = new ArrayList<>();
    for (File candidate : candidates) {
      try {
        String content = readCandidate(candidate);
        LOGGER.fine(
            () ->
                "Type-annotation sidecar candidate " + candidate + ": " + (content != null) + ".");
        if (content == null) continue;
        JSONObject sidecar = new JSONObject(content);
        JSONArray types = sidecar.optJSONArray("types");
        if (types == null) {
          LOGGER.warning("Type-annotation sidecar " + candidate + " has no types array; skipped.");
          continue;
        }
        for (int i = 0; i < types.length(); i++) {
          Entry entry = parseEntry(types.getJSONObject(i), candidate);
          if (entry != null) ret.add(entry);
        }
      } catch (IOException | JSONException e) {
        LOGGER.warning("Unreadable type-annotation sidecar " + candidate + ": " + e + "; skipped.");
      }
    }
    LOGGER.fine(() -> "Loaded " + ret.size() + " type-annotation sidecar entries.");
    return ret;
  }

  /**
   * Reads a sidecar candidate's content: directly for an ordinary file, and through a {@code jar:}
   * URL when the Python-path root is a packaged resource (the in-suite harness resolves fixture
   * projects from the test-data jar).
   *
   * @param candidate The candidate sidecar file.
   * @return The file content, or {@code null} when the candidate does not exist.
   * @throws IOException On a read error of an existing candidate.
   */
  private static String readCandidate(File candidate) throws IOException {
    if (candidate.isFile()) return Files.readString(candidate.toPath());
    String path = candidate.getPath();
    if (path.contains(".jar!")) {
      String spec = "jar:" + (path.startsWith("file:") ? path : "file:" + path);
      try (java.io.InputStream in = java.net.URI.create(spec).toURL().openStream()) {
        return new String(in.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
      } catch (IOException e) {
        return null; // No such jar entry.
      }
    }
    return null;
  }

  /**
   * Parses one sidecar entry, reporting and skipping malformed ones.
   *
   * @param object The entry's JSON object.
   * @param file The sidecar file, for reports.
   * @return The entry, or {@code null} when malformed.
   */
  private static Entry parseEntry(JSONObject object, File file) {
    String module = object.optString("module", "");
    String variable = object.optString("variable", "");
    if (module.isEmpty() || variable.isEmpty()) {
      LOGGER.warning(
          "Type-annotation entry in " + file + " lacks module or variable: " + object + ".");
      return null;
    }
    String function = object.optString("function", "");
    String dtypeName = object.has("dtype") ? object.getString("dtype") : null;
    DType dtype = DType.UNKNOWN;
    if (dtypeName != null) {
      try {
        dtype = DType.valueOf(dtypeName.toUpperCase(Locale.ROOT));
      } catch (IllegalArgumentException e) {
        LOGGER.warning(
            "Type-annotation entry in " + file + " names unknown dtype " + dtypeName + ".");
        return null;
      }
    }
    List<Dimension<?>> dims = null;
    if (object.has("shape")) {
      dims = parseDimensions(object.getString("shape"));
      if (dims == null && !"...".equals(object.getString("shape").trim())) {
        LOGGER.warning(
            "Type-annotation entry in "
                + file
                + " has an unparseable shape "
                + object.getString("shape")
                + ".");
        return null;
      }
    }
    return new Entry(module, function, variable, new TensorType(dtype, dims));
  }

  /**
   * Parses a dimension string into dimensions: whitespace-separated integers ({@link NumericDim})
   * or names ({@link SymbolicDim}); {@code "..."} alone means unknown rank; the empty string means
   * rank 0.
   *
   * @param shape The dimension string.
   * @return The dimensions, or {@code null} for unknown rank.
   */
  private static List<Dimension<?>> parseDimensions(String shape) {
    String trimmed = shape.trim();
    if ("...".equals(trimmed)) return null;
    List<Dimension<?>> dims = new ArrayList<>();
    if (trimmed.isEmpty()) return dims;
    for (String token : trimmed.split("\\s+")) {
      try {
        dims.add(new NumericDim(Integer.parseInt(token)));
      } catch (NumberFormatException e) {
        if (!token.matches("[A-Za-z_][A-Za-z0-9_]*")) return null;
        dims.add(new SymbolicDim(token));
      }
    }
    return dims;
  }
}
