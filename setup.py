import os
import tarfile
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Distutils import extension
import numpy as np
import urllib

qpbo_directory = "QPBO-v1.3.src"
maxflow_dir = "LSA/maxflow-v3.01/"
files = ["QPBO.cpp","QPBO_maxflow.cpp",
         "QPBO_postprocessing.cpp"]

files = [os.path.join(qpbo_directory, f) for f in files]
files.insert(0, "_helper.pyx")
files.append(maxflow_dir+"graph.cpp")
files.append(maxflow_dir+"maxflow.cpp")

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('helper', sources=files, language='c++',
                    include_dirs=[qpbo_directory, np.get_include(), "LSA", "LSA/QPBO-v1.3.src", "LSA/maxflow-v3.01", "LSA/Eigen"],                    
                    library_dirs=[qpbo_directory],
                    extra_compile_args=["-std=c++0x", "-O2", "-msse2", "-fpermissive",'-fopenmp'],                    
                                              extra_link_args=['-fopenmp'])
        ]
    )

