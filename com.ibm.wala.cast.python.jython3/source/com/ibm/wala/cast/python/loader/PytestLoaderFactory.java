package com.ibm.wala.cast.python.loader;

import com.ibm.wala.classLoader.IClassLoader;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import java.io.File;
import java.util.List;

public class PytestLoaderFactory extends PythonLoaderFactory {

  protected List<File> pythonPath;

  public PytestLoaderFactory(List<File> pythonPath) {
    this.pythonPath = pythonPath;
  }

  @Override
  protected IClassLoader makeTheLoader(IClassHierarchy cha) {
    return new PytestLoader(cha, this.getPythonPath());
  }

  public List<File> getPythonPath() {
    return pythonPath;
  }
}
