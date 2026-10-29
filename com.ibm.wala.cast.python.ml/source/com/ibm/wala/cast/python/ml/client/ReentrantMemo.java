package com.ibm.wala.cast.python.ml.client;

import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Supplier;

/**
 * A memo whose computation may re-enter itself on the same key, safe to share between threads (<a
 * href="https://github.com/wala/ML/issues/1175">wala/ML#1175</a>). A computation that meets its own
 * key again on the same thread reads the floor, {@link Optional#empty()}, instead of recursing,
 * which is what a sentinel stored in the memo before computing did. The mark is per thread, so a
 * thread that reads a key another thread is still computing computes it itself rather than taking
 * the other thread's floor as the answer. Each thread floors only its own re-entries, as the one
 * thread of an analysis run always has, so a single-threaded run memoizes exactly what the sentinel
 * did.
 *
 * @param <V> The type of a computed value.
 */
final class ReentrantMemo<V> {

  /** The values computed so far, an absent one stored as {@link Optional#empty()}. */
  private final Map<Object, Optional<V>> values;

  /** The keys the current thread is computing. */
  private final ThreadLocal<Set<Object>> inProgress = ThreadLocal.withInitial(HashSet::new);

  /**
   * Creates a memo over the given map.
   *
   * @param values The map holding the computed values, which must be safe to share between threads
   *     and put a value if absent atomically.
   */
  ReentrantMemo(Map<Object, Optional<V>> values) {
    this.values = values;
  }

  /**
   * Returns the value memoized for the given key, computing and storing it when there is none. A
   * computation that re-enters the same key on the same thread reads {@link Optional#empty()},
   * which is not stored. When threads compute the same key at once, the first value stored stands,
   * and every one of them returns it, so all reads of a key agree.
   *
   * @param key The key.
   * @param compute The computation, returning {@code null} for an absent value.
   * @return The value, or {@link Optional#empty()} for an absent one or a re-entry.
   */
  Optional<V> get(Object key, Supplier<V> compute) {
    Optional<V> memo = this.values.get(key);
    if (memo != null) return memo;
    Set<Object> active = this.inProgress.get();
    if (!active.add(key)) return Optional.empty();
    try {
      memo = Optional.ofNullable(compute.get());
      Optional<V> prior = this.values.putIfAbsent(key, memo);
      return prior != null ? prior : memo;
    } finally {
      active.remove(key);
    }
  }
}
