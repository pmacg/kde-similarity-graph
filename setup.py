"""
Compile FSG.
"""
import platform
import numpy
import os
from distutils.core import setup, Extension

if platform.system() == 'Linux':
    # compile_args = ['-std=c++2a', '-fopenmp']
    compile_args = ['-std=c++2a']
    include_dirs = [
        "src/cpp/fgt/include",
        "src/cpp/stag_lib",
        "src/cpp/stag_lib/KMeansRex",
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "src/cpp/fgt/vendor/nanoflann-1.1.9/include",
        os.path.join(numpy.__path__[0], 'core/include')
    ]
elif platform.system() == 'Windows':
    compile_args = ['/std:c++20']
    include_dirs = ["src/cpp/fgt/include",
                    "src/cpp/stag_lib",
                    "src/cpp/stag_lib/KMeansRex",
                    # Add location of EIGEN and SPECTRA libraries here!
                    "src/cpp/fgt/vendor/nanoflann-1.1.9/include",
                    os.path.join(numpy.__path__[0], 'core\\include')]
else:
    compile_args = ['-std=c++2a']
    include_dirs = [
        "src/cpp/fgt/include",
        "src/cpp/stag_lib",
        "src/cpp/stag_lib/KMeansRex",
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "src/cpp/fgt/vendor/nanoflann-1.1.9/include",
        os.path.join(numpy.__path__[0], 'core/include')
    ]

# link_args = ['-lgomp']
link_args = []

fsg_module = Extension('src._fsg_internal',
                       sources=[
                           'src/cpp/stag_lib/utility.cpp',
                           'src/cpp/stag_lib/graph.cpp',
                           'src/cpp/stag_lib/cluster.cpp',
                           'src/cpp/stag_lib/random.cpp',
                           'src/cpp/stag_lib/spectrum.cpp',
                           'src/cpp/fsg.cpp',
                           'src/cpp/kde.cpp',
                           'src/cpp/utilities.cpp',
                           'src/cpp/stag_lib/KMeansRex/KMeansRexCore.cpp',
                           'src/cpp/stag_lib/KMeansRex/mersenneTwister2002.c',
                           'src/cpp/fgt/src/cluster.cpp',
                           'src/cpp/fgt/src/cluster-openmp.cpp',
                           'src/cpp/fgt/src/direct.cpp',
                           'src/cpp/fgt/src/direct_tree.cpp',
                           'src/cpp/fgt/src/ifgt.cpp',
                           'src/cpp/fgt/src/openmp.cpp',
                           'src/cpp/fgt/src/transform.cpp',
                           'src/fsg_internal_wrap.cxx'
                       ],
                       include_dirs=include_dirs,
                       extra_compile_args=compile_args,
                       extra_link_args=link_args)


setup(name="fsg",
      version='0.1',
      author="Anonymous",
      description="Fast Similarity Graph",
      ext_modules=[fsg_module],
      py_modules=["src/fsg"])
