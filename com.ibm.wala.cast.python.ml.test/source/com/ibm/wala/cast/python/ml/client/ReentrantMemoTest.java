package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;

/**
 * A memo whose computation may re-enter itself floors a re-entry on the computing thread only (<a
 * href="https://github.com/wala/ML/issues/1175">wala/ML#1175</a>).
 */
public class ReentrantMemoTest {

  /**
   * A computation that re-enters its own key reads the floor, the floor isn't stored, and the
   * computed value is.
   */
  @Test
  public void testReentryOnTheSameThreadReadsTheFloor() {
    Map<Object, Optional<String>> values = Collections.synchronizedMap(new HashMap<>());
    ReentrantMemo<String> memo = new ReentrantMemo<>(values);
    AtomicReference<Optional<String>> inner = new AtomicReference<>();
    Optional<String> outer =
        memo.get(
            "k",
            () -> {
              inner.set(memo.get("k", () -> "recursed"));
              return "computed";
            });
    assertEquals(Optional.empty(), inner.get());
    assertEquals(Optional.of("computed"), outer);
    assertEquals(Optional.of("computed"), values.get("k"));
  }

  /**
   * A thread that reads a key while another thread is computing it computes the key itself rather
   * than reading the other thread's floor. A latch holds the first thread inside its computation,
   * after the key is marked and before the value is stored, while the second thread reads.
   */
  @Test
  public void testAnotherThreadDoesNotReadTheFloor() throws Exception {
    ReentrantMemo<String> memo = new ReentrantMemo<>(Collections.synchronizedMap(new HashMap<>()));
    CountDownLatch entered = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    ExecutorService first = Executors.newSingleThreadExecutor();
    try {
      Future<Optional<String>> held =
          first.submit(
              () ->
                  memo.get(
                      "k",
                      () -> {
                        entered.countDown();
                        try {
                          assertTrue(release.await(30, TimeUnit.SECONDS));
                        } catch (InterruptedException e) {
                          Thread.currentThread().interrupt();
                        }
                        return "first";
                      }));
      assertTrue("The first thread entered its computation.", entered.await(30, TimeUnit.SECONDS));
      assertEquals(Optional.of("second"), memo.get("k", () -> "second"));
      release.countDown();
      assertEquals(Optional.of("first"), held.get(30, TimeUnit.SECONDS));
    } finally {
      release.countDown();
      first.shutdownNow();
    }
  }

  /** A computation that throws leaves its key unmarked, so a later read computes it. */
  @Test
  public void testAThrowingComputationLeavesTheKeyUnmarked() {
    ReentrantMemo<String> memo = new ReentrantMemo<>(Collections.synchronizedMap(new HashMap<>()));
    try {
      memo.get(
          "k",
          () -> {
            throw new IllegalStateException();
          });
      fail("The computation's exception propagates.");
    } catch (IllegalStateException expected) {
      // As thrown.
    }
    Optional<String> later = memo.get("k", () -> "later");
    assertFalse(later.isEmpty());
    assertEquals(Optional.of("later"), later);
  }
}
