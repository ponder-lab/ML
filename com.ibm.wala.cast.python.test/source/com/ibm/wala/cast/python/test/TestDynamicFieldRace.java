package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Test;

/**
 * A Python class's fields stay consistent while other threads look up names it does not have. A
 * miss creates the field on the root class, so lookups from a client's parallel threads (such as a
 * consumer checking functions in parallel) add fields while other threads enumerate them. With the
 * root's plain map, an enumeration could throw a {@link java.util.ConcurrentModificationException}
 * or, during a resize, skip fields that existed all along; a client deciding from those fields
 * would then decide differently from run to run.
 */
public class TestDynamicFieldRace extends TestJythonCallGraphShape {

  private static final int EXISTING = 2_000;

  private static final int THREADS = 4;

  private static final int MISSES_PER_THREAD = 20_000;

  /**
   * Enumerations of a class's instance fields that race lookups of fresh names neither throw nor
   * leave out a field that existed before the race.
   *
   * @throws Exception if the class hierarchy cannot be built or a thread is interrupted.
   */
  @Test
  public void testEnumerationDuringMissesKeepsExistingFields() throws Exception {
    IClassHierarchy cha = makeEngine("classes1.py").defaultCallGraphBuilder().getClassHierarchy();
    IClass object = cha.lookupClass(PythonTypes.object);
    assertNotNull("No class for `object`.", object);

    for (int i = 0; i < EXISTING; i++)
      object.getField(Atom.findOrCreateUnicodeAtom("existing" + i));
    Set<IField> before = new HashSet<>(object.getAllInstanceFields());
    assertTrue(before.size() >= EXISTING);

    CountDownLatch start = new CountDownLatch(1);
    AtomicBoolean writing = new AtomicBoolean(true);
    ConcurrentLinkedQueue<Throwable> failures = new ConcurrentLinkedQueue<>();
    ConcurrentLinkedQueue<String> omissions = new ConcurrentLinkedQueue<>();
    List<Thread> writers = new ArrayList<>();
    List<Thread> readers = new ArrayList<>();
    for (int t = 0; t < THREADS; t++) {
      int thread = t;
      writers.add(
          new Thread(
              () -> {
                try {
                  start.await();
                  for (int i = 0; i < MISSES_PER_THREAD; i++)
                    object.getField(Atom.findOrCreateUnicodeAtom("miss" + thread + "_" + i));
                } catch (Throwable e) {
                  failures.add(e);
                }
              }));
      readers.add(
          new Thread(
              () -> {
                try {
                  start.await();
                  while (writing.get()) {
                    Collection<IField> fields = object.getAllInstanceFields();
                    if (!fields.containsAll(before))
                      omissions.add(
                          (before.size() - before.stream().filter(fields::contains).count())
                              + " existing fields left out");
                  }
                } catch (Throwable e) {
                  failures.add(e);
                }
              }));
    }
    writers.forEach(Thread::start);
    readers.forEach(Thread::start);
    start.countDown();
    for (Thread writer : writers) writer.join();
    writing.set(false);
    for (Thread reader : readers) reader.join();

    assertTrue("Enumeration failed: " + failures, failures.isEmpty());
    assertTrue("Enumeration left out existing fields: " + omissions, omissions.isEmpty());

    // Every name looked up resolves to one field, the same on every lookup.
    Collection<IField> after = object.getAllInstanceFields();
    for (int t = 0; t < THREADS; t++)
      for (int i = 0; i < MISSES_PER_THREAD; i += 997) {
        Atom name = Atom.findOrCreateUnicodeAtom("miss" + t + "_" + i);
        IField field = object.getField(name);
        assertSame(field, object.getField(name));
        assertTrue(after.contains(field));
      }
    assertEquals(
        before.size() + THREADS * MISSES_PER_THREAD,
        after.stream().filter(f -> !f.isStatic()).count());
  }
}
