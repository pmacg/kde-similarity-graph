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
        "fsg/cpp/fgt/include",
        "fsg/cpp/stag_lib",
        "fsg/cpp/stag_lib/KMeansRex",
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "fsg/cpp/fgt/vendor/nanoflann-1.1.9/include",
        os.path.join(numpy.__path__[0], 'core/include')
    ]
elif platform.system() == 'Windows':
    compile_args = ['/std:c++20']
    include_dirs = ["fsg/cpp/fgt/include",
                    "fsg/cpp/stag_lib",
                    "fsg/cpp/stag_lib/KMeansRex",
                    "C:/Users/macgr/wc/vcpkg/packages/eigen3_x86-windows/include",
                    "C:/Users/macgr/wc/vcpkg/packages/spectra_x86-windows/include",
                    "fsg/cpp/fgt/vendor/nanoflann-1.1.9/include",
                    os.path.join(numpy.__path__[0], 'core\\include')]
else:
    compile_args = ['-std=c++2a']
    include_dirs = [
        "fsg/cpp/fgt/include",
        "fsg/cpp/stag_lib",
        "fsg/cpp/stag_lib/KMeansRex",
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "fsg/cpp/fgt/vendor/nanoflann-1.1.9/include",
        os.path.join(numpy.__path__[0], 'core/include')
    ]

# link_args = ['-lgomp']
link_args = []

fsg_module = Extension('fsg._fsg_internal',
                       sources=[
                           'fsg/cpp/stag_lib/utility.cpp',
                           'fsg/cpp/stag_lib/graph.cpp',
                           'fsg/cpp/stag_lib/cluster.cpp',
                           'fsg/cpp/stag_lib/random.cpp',
                           'fsg/cpp/stag_lib/spectrum.cpp',
                           'fsg/cpp/fsg.cpp',
                           'fsg/cpp/kde.cpp',
                           'fsg/cpp/stag_lib/KMeansRex/KMeansRexCore.cpp',
                           'fsg/cpp/stag_lib/KMeansRex/mersenneTwister2002.c',
                           'fsg/cpp/fgt/src/cluster.cpp',
                           'fsg/cpp/fgt/src/cluster-openmp.cpp',
                           'fsg/cpp/fgt/src/direct.cpp',
                           'fsg/cpp/fgt/src/direct_tree.cpp',
                           'fsg/cpp/fgt/src/ifgt.cpp',
                           'fsg/cpp/fgt/src/openmp.cpp',
                           'fsg/cpp/fgt/src/transform.cpp',
                           'fsg/fsg_internal_wrap.cxx'
                       ],
                       include_dirs=include_dirs,
                       extra_compile_args=compile_args,
                       extra_link_args=link_args)


setup(name="fsg",
      version='0.1',
      author="Anonymous",
      description="Fast Similarity Graph",
      ext_modules=[fsg_module],
      py_modules=["fsg/fsg"])
