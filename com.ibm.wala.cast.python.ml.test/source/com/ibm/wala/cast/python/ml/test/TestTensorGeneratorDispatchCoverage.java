package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.fail;

import com.ibm.wala.cast.python.ml.client.DispatchExempt;
import com.ibm.wala.cast.python.ml.client.TensorGenerator;
import java.io.IOException;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.Test;

/**
 * Meta-test asserting that every concrete {@link TensorGenerator} subclass under {@code
 * com.ibm.wala.cast.python.ml.client} is reachable from at least one of the two dispatch tables —
 * {@link com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory#getGenerator} or {@link
 * TensorGenerator#createManualGenerator}.
 *
 * <p>Until the dispatch-table unification proposed in wala/ML#469 lands, every new {@code
 * TensorGenerator} subclass has to be added to both {@code getGeneratorBody} (factory-side) and
 * {@code createManualGenerator} (manual-walker side). Forgetting one is silent (wala/ML#468). This
 * coverage guard catches the most-common failure mode — an entirely orphan subclass that's not
 * wired to either dispatch — at test time. (The stricter "appears in <em>both</em>" check the
 * issue's original spec proposes requires either dynamic dispatch construction or per-class
 * annotation tagging for the dispatch-asymmetric subclasses; this test scopes to the orphan
 * detection that's implementable without those.)
 *
 * <p>Subclasses that are legitimately exempt (constructed by another generator rather than from
 * either dispatch table) can carry a {@link DispatchExempt} annotation. Abstract bases are skipped
 * automatically.
 *
 * @see DispatchExempt
 * @see <a href="https://github.com/wala/ML/issues/470">wala/ML#470</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestTensorGeneratorDispatchCoverage {

  private static final String CLIENT_PACKAGE = "com.ibm.wala.cast.python.ml.client";

  private static final String CLIENT_PACKAGE_PATH = CLIENT_PACKAGE.replace('.', '/');

  /**
   * Asserts every concrete, non-exempt {@link TensorGenerator} subclass under {@link
   * #CLIENT_PACKAGE} appears as a {@code new <SimpleName>(} construction in at least one of the two
   * dispatch source files.
   *
   * @throws Exception If reflection or file I/O fails.
   */
  @Test
  public void everyConcreteTensorGeneratorAppearsInAtLeastOneDispatch() throws Exception {
    List<Class<? extends TensorGenerator>> subclasses = enumerateTensorGeneratorSubclasses();

    String factorySource = readDispatchSource("TensorGeneratorFactory.java");
    String tensorGeneratorSource = readDispatchSource("TensorGenerator.java");

    List<String> orphans = new ArrayList<>();
    for (Class<? extends TensorGenerator> c : subclasses) {
      if (Modifier.isAbstract(c.getModifiers())) continue;
      if (c.isAnnotationPresent(DispatchExempt.class)) continue;
      // Skip anonymous and synthetic inner classes — they're inline-declared and inherently can't
      // be wired to a separate dispatch table.
      if (c.getCanonicalName() == null) continue;
      if (c.isAnonymousClass() || c.isSynthetic()) continue;

      String simpleName = c.getSimpleName();
      // Tolerate spotless/IDE formatting variants — `new X(...)`, `new\n        X(...)`, etc.
      Pattern constructionPattern =
          Pattern.compile("\\bnew\\s+" + Pattern.quote(simpleName) + "\\s*\\(");
      boolean inFactory = constructionPattern.matcher(factorySource).find();
      boolean inTensorGenerator = constructionPattern.matcher(tensorGeneratorSource).find();

      if (!inFactory && !inTensorGenerator) {
        orphans.add(simpleName);
      }
    }

    if (!orphans.isEmpty()) {
      fail(
          "The following concrete TensorGenerator subclasses are not constructed in either"
              + " TensorGeneratorFactory.getGenerator or TensorGenerator.createManualGenerator. If"
              + " a subclass is intentionally constructed only via delegation by another"
              + " generator, mark it with @DispatchExempt. Otherwise it's an orphan — wire it"
              + " into the appropriate dispatch table.\n  - "
              + String.join("\n  - ", orphans));
    }
  }

  /**
   * Walks the ml module's source tree under {@link #CLIENT_PACKAGE} (recursively, including any
   * future subpackages), deriving the FQN from each {@code .java} file's path relative to the
   * source root and loading it via {@link Class#forName} <em>without</em> initialization (so static
   * initializers in any subclass don't fire as a side effect of the meta-test). Returns the loaded
   * classes that are (non-{@link TensorGenerator}-itself) subclasses of {@link TensorGenerator}.
   *
   * <p>Walks the source tree (rather than the test classpath) because the ml module is packaged as
   * a JAR on the test classpath, and {@link Paths#get(java.net.URI)} on a {@code jar:} URI requires
   * opening the JAR file system explicitly. Walking the source tree avoids that complication and is
   * independent of how the dependency is packaged.
   *
   * @return The concrete and abstract {@link TensorGenerator} subclasses found.
   * @throws IOException If walking the directory fails.
   */
  private static List<Class<? extends TensorGenerator>> enumerateTensorGeneratorSubclasses()
      throws IOException {
    Path mlSourceRoot = locateMlSourceRoot();
    Path packageDir = mlSourceRoot.resolve(CLIENT_PACKAGE_PATH);
    if (!Files.isDirectory(packageDir))
      throw new IllegalStateException(
          "Could not locate ml-module source package directory: " + packageDir);

    List<Class<? extends TensorGenerator>> subclasses = new ArrayList<>();
    ClassLoader cl = TestTensorGeneratorDispatchCoverage.class.getClassLoader();
    try (Stream<Path> entries = Files.walk(packageDir)) {
      entries
          .filter(Files::isRegularFile)
          .filter(p -> p.getFileName().toString().endsWith(".java"))
          .forEach(
              p -> {
                Path relative = mlSourceRoot.relativize(p);
                String className =
                    relative
                        .toString()
                        .replace(java.io.File.separatorChar, '.')
                        .replaceFirst("\\.java$", "");
                try {
                  Class<?> c = Class.forName(className, false, cl);
                  if (TensorGenerator.class.isAssignableFrom(c)
                      && !TensorGenerator.class.equals(c)) {
                    subclasses.add(c.asSubclass(TensorGenerator.class));
                  }
                } catch (ClassNotFoundException e) {
                  throw new IllegalStateException(
                      "Failed to load " + className + " — should be on the test classpath.", e);
                }
              });
    }
    return subclasses;
  }

  /**
   * Reads a source file from the {@code com.ibm.wala.cast.python.ml/source/...} module relative to
   * the located ml-module source root.
   *
   * @param fileName The simple file name (e.g., {@code TensorGenerator.java}).
   * @return The file contents.
   * @throws IOException If the file can't be read.
   */
  private static String readDispatchSource(String fileName) throws IOException {
    Path src = locateMlSourceRoot().resolve(CLIENT_PACKAGE_PATH).resolve(fileName);
    if (!Files.exists(src))
      throw new IllegalStateException("Could not locate dispatch source: " + src);
    return Files.readString(src);
  }

  /**
   * Locates the ml module's source root ({@code com.ibm.wala.cast.python.ml/source/}) by trying
   * common Maven/IDE working-directory anchors, walking up the filesystem if necessary. Resilient
   * to running from the test module directly (Maven's default), the parent project root (some IDE
   * configurations), or other sibling locations.
   *
   * @return The {@link Path} of the ml source root.
   */
  private static Path locateMlSourceRoot() {
    Path cwd = Paths.get(System.getProperty("user.dir"));
    for (Path candidate = cwd; candidate != null; candidate = candidate.getParent()) {
      Path direct = candidate.resolve("com.ibm.wala.cast.python.ml").resolve("source");
      if (Files.isDirectory(direct)) return direct;
      // If we're sitting in the test module itself, the ml module is a sibling.
      Path sibling = candidate.resolveSibling("com.ibm.wala.cast.python.ml").resolve("source");
      if (Files.isDirectory(sibling)) return sibling;
    }
    throw new IllegalStateException(
        "Could not locate ml-module source root from cwd=" + cwd + " — check working directory.");
  }
}
