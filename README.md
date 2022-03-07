# Push-Relabel Based Additive Approximation for Optimal Transport Algorithm Implementation

This repository contains the implementation and experiment code for the optimal transport described on [arXiv](). This is a joint work of the paper authors and [Abhijeet Phatak](https://github.com/abhijit-15).

Optimal Transport (OT) is a metric measuring the similarity between distributions. In discrete OT, we are given the point sets <img src="https://latex.codecogs.com/gif.latex?A" /> and <img src="https://latex.codecogs.com/gif.latex?B" />, which can be viewed as a unit amount of earth piled on a given metric space <img src="https://latex.codecogs.com/gif.latex?M" />. The metric is the minimum effort moving one pile to another, which is known as earth mover's distance(EMD). Now, this metric is popularly used in the machine learning community.

Exact algorithms for computing OT can be slow, which has motivated the development of approximate numerical solvers (e.g. Sinkhorn method). We introduce a new and very simple combinatorial approach to find an <img src="https://latex.codecogs.com/gif.latex?\varepsilon" />-approximation of the OT distance. Our algorithm achieves a near-optimal execution time of <img src="https://latex.codecogs.com/gif.latex?O(n^2/\varepsilon^2)" /> for computing OT distance and, for the special case of the assignment problem, the execution time improves to <img src="https://latex.codecogs.com/gif.latex?O(n^2/\varepsilon)" />. Our algorithm is based on the push-relabel framework for min-cost flow problems.

So far, we have completed the implementation of the assignment algorithm. In this case, <img src="https://latex.codecogs.com/gif.latex?A" /> and <img src="https://latex.codecogs.com/gif.latex?B" /> each contain <img src="https://latex.codecogs.com/gif.latex?n" /> points and every point in <img src="https://latex.codecogs.com/gif.latex?A" /> (resp. <img src="https://latex.codecogs.com/gif.latex?B" />) has a demand of <img src="https://latex.codecogs.com/gif.latex?1/n" /> (resp. supply of <img src="https://latex.codecogs.com/gif.latex?1/n" />). We provide a CPU and a GPU implementation, and in the experiment, our algorithm is faster than the Sinkhorn algorithm both in CPU and GPU implementation. 

The OT implementation is a work in progress.



## Requirements
This repository contains two parts: 

1. Implementation of assignment algorithm: `matching.py`
2. Experiments compare our method and Sinkhorn: `pl_vs_sinkorn_mnist.py`,`pl_vs_sinkorn_synthetic.py`

### Dependencies

To use our algorithm or reproduce our experiments, simply install the following dependencies in your python environment and run the code.

For the first part, our algorithm implmentaion requires:

- [NumPy](https://numpy.org/install/) v1.21 
- [PyTorch](https://pytorch.org/) v1.10

Reproducing our experiments requires:

- [CUDA Toolkit](https://developer.nvidia.com/cuda-11.3.0-download-archive) v11.3
- [CuPy](https://docs.cupy.dev/en/stable/install.html) v9.6
- [NumPy](https://numpy.org/install/) v1.21
- [POT](https://pythonot.github.io/) v0.8.1
- [PyTorch](https://pytorch.org/) v1.10
- [TensorFlow](https://www.tensorflow.org/install) v2.6


## 