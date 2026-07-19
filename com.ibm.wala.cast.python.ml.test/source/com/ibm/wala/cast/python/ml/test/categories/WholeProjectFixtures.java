package com.ibm.wala.cast.python.ml.test.categories;

/**
 * JUnit category marking tests that analyze a whole vendored project (the suite's long pole), so CI
 * can run them in a job parallel to the rest of the suite (wala/ML#755). Annotate the test class
 * with {@code @Category(WholeProjectFixtures.class)}.
 */
public interface WholeProjectFixtures {}
