import os
import tarfile
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Distutils import extension
import numpy as np
import urllib

qpbo_directory = "QPBO-v1.3.src"
files = ["QPBO.cpp","QPBO_maxflow.cpp",
         "QPBO_postprocessing.cpp"]

files = [os.path.join(qpbo_directory, f) for f in files]
files.insert(0, "_helper.pyx")
# fastpd_dir = "fastpd/src/"
# for f in os.listdir(fastpd_dir):
#     if f.endswith("cpp"):
#         files.append(fastpd_dir + f)

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('helper', sources=files, language='c++',
                    include_dirs=[ np.get_include(), qpbo_directory],                    
                    library_dirs=[],
                    extra_compile_args=["-std=c++0x","-O2", "-msse2",'-fopenmp', "-fpermissive"],                    
                                              extra_link_args=['-fopenmp'])
        ]
    )

