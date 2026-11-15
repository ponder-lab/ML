package com.ibm.wala.cast.python.ipa.callgraph;

/**
 * A call site applying a decorator to the class object it decorates (<a
 * href="https://github.com/wala/ML/issues/1260">wala/ML#1260</a>), marked so the call graph builder
 * can tell a class decorator's application from any other call.
 */
public interface ClassDecoratorSite {}
