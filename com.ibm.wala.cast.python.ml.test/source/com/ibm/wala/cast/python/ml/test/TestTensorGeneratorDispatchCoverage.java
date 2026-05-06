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
      String constructionPattern = "new " + simpleName + "(";
      boolean inFactory = factorySource.contains(constructionPattern);
      boolean inTensorGenerator = tensorGeneratorSource.contains(constructionPattern);

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
   * Walks the ml module's source directory for {@link #CLIENT_PACKAGE}, deriving the class name
   * from each {@code .java} filename and loading it via {@link Class#forName}. Returns the loaded
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
    Path testModuleDir = Paths.get(System.getProperty("user.dir"));
    Path mlModuleDir = testModuleDir.resolveSibling("com.ibm.wala.cast.python.ml");
    Path packageDir = mlModuleDir.resolve("source").resolve(CLIENT_PACKAGE_PATH);
    if (!Files.isDirectory(packageDir))
      throw new IllegalStateException(
          "Could not locate ml-module source package directory: " + packageDir);

    List<Class<? extends TensorGenerator>> subclasses = new ArrayList<>();
    try (Stream<Path> entries = Files.list(packageDir)) {
      entries
          .filter(p -> p.getFileName().toString().endsWith(".java"))
          .forEach(
              p -> {
                String fileName = p.getFileName().toString();
                String className =
                    CLIENT_PACKAGE
                        + "."
                        + fileName.substring(0, fileName.length() - ".java".length());
                try {
                  Class<?> c = Class.forName(className);
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
   * Reads a source file from the {@code com.ibm.wala.cast.python.ml/source/...} module by relative
   * path from the test module's working directory.
   *
   * @param fileName The simple file name (e.g., {@code TensorGenerator.java}).
   * @return The file contents.
   * @throws IOException If the file can't be read.
   */
  private static String readDispatchSource(String fileName) throws IOException {
    Path testModuleDir = Paths.get(System.getProperty("user.dir"));
    Path mlModuleDir = testModuleDir.resolveSibling("com.ibm.wala.cast.python.ml");
    Path src = mlModuleDir.resolve("source").resolve(CLIENT_PACKAGE_PATH).resolve(fileName);
    if (!Files.exists(src))
      throw new IllegalStateException("Could not locate dispatch source: " + src);
    return Files.readString(src);
  }
}
