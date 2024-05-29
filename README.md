# Fast Approximation of Similarity Graphs with Kernel Density Estimation.
This repository contains code to accompany the paper "Fast Approximation of Similarity Graphs with Kernel Density Estimation",
published at NeurIPS'23.

## Build Instructions

The similarity graph construction code is written in C++, in the `src/cpp/` directory.
There is then a python wrapper around this C++ code.
To compile the code, follow the instructions below.
It is recommended to use a Python conda environment, as this is the easiest way to install the project dependencies.

### Install the C++ dependencies

The C++ code requires the following libraries to be installed.
- Eigen
- Spectra

You should refer to their documentation for installation instructions, although
the following should work on a standard linux system.

```bash
# Create a directory to work in
mkdir libraries
cd libraries

# Install Eigen
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xzvf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build_dir
cd build_dir
cmake ..
sudo make install
cd ../..

# Install Spectra
wget https://github.com/yixuan/spectra/archive/v1.0.1.tar.gz
tar xzvf v1.0.1.tar.gz
cd spectra-1.0.1
mkdir build_dir
cd build_dir
cmake ..
sudo make install
cd ../..
```

### Compile the C++ Python extension
First, create a new conda envorinment.

```bash
conda create --name my-env
conda activate my-env
```

Then, in the root directory (the one containing this README file), run the following
commands:

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

This will compile the C++ code and create an extension file which can be imported
by the Python code.

### Install the FAISS library
We compare our algorithm against the approximate nearest neighbour graphs constructed
with the FAISS library. Install FAISS with conda.

```bash
conda install -c pytorch faiss-cpu
```

## Running the experiments

There are three experiments reported in the paper.

### Clustering comparison

To run the experiments for and create a figure like Figure 1,
run the `sklearn_examples.py` script:

```
python3 fsg/sklearn_examples.py
```

Note that there is a parameter at the top of the script which allows you to select
the number of data points to generate in each dataset.

### Two moons experiment

To run the experiment for comparing the algorithms' running time on the two
moons dataset, run the following conda command.

```bash
conda run --no-capture-output python main.py run moons
```

Then, to create the figures as reproduced in the paper, run the following.

```bash
conda run --no-capture-output python main.py plot moons
```

### BSDS experiment
To run the experiment for comparing the algorithms' performance on the image
segmentation task, first ensure that the BSDS data is extracted in the `data` directory.

Then, run the following conda command.

```bash
conda run --no-capture-output python main.py run bsds
```

Note that the full BSDS experiment takes a long time to run.
You can also run the experiment on a single image ID with

```bash
conda run --no-capture-output python main.py run bsds --id {id}
```

Then, to create the figures as reproduced in the paper, run the following.

```bash
conda run --no-capture-output python main.py plot bsds --id {id}
```

where `{id}` is the ID of the BSDS image you would like to plot.
The IDs plotted in the paper are 35049, 208078, 61060, 135069, 2018, and 181021.
