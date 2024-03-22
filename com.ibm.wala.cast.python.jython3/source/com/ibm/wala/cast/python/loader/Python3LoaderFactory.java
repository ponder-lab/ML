package com.ibm.wala.cast.python.loader;

import com.ibm.wala.classLoader.IClassLoader;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import java.io.File;
import java.util.List;

public class Python3LoaderFactory extends PythonLoaderFactory {

  protected List<File> pythonPath;

  public Python3LoaderFactory(List<File> pythonPath) {
    this.pythonPath = pythonPath;
  }

  @Override
  protected IClassLoader makeTheLoader(IClassHierarchy cha) {
    return new Python3Loader(cha, this.getPythonPath());
  }

  public List<File> getPythonPath() {
    return pythonPath;
  }
}
