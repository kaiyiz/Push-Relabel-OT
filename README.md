# Push-Relabel Based Additive Approximation for Optimal Transport Algorithm Implementation

This repository contains the implementation and experiment code for the paper titled "A Push-Relabel Based Additive Approximation for Optimal Transport", which is available on [arXiv](https://arxiv.org/abs/2203.03732). This paper introduces algorithms for the Optimal Transport (OT) problem and the assignment problem.

Optimal Transport (OT) is a metric measuring the similarity between distributions. In discrete OT, we are given discrete probability distributions on point sets <img src="https://latex.codecogs.com/gif.latex?A" /> and <img src="https://latex.codecogs.com/gif.latex?B" />, each of size <img src="https://latex.codecogs.com/gif.latex?n" />. We are also given an <img src="https://latex.codecogs.com/gif.latex?n" /> <img src="https://latex.codecogs.com/gif.latex?\times" /> <img src="https://latex.codecogs.com/gif.latex?n" /> cost matrix <img src="https://latex.codecogs.com/gif.latex?W" />, where <img src="https://latex.codecogs.com/gif.latex?W(a,b)" /> gives the cost of transporting one unit of mass from the point <img src="https://latex.codecogs.com/gif.latex?b" /> <img src="https://latex.codecogs.com/gif.latex?\in" /> <img src="https://latex.codecogs.com/gif.latex?B" /> to the point <img src="https://latex.codecogs.com/gif.latex?a" /> <img src="https://latex.codecogs.com/gif.latex?\in" /> <img src="https://latex.codecogs.com/gif.latex?A" />. The OT distance is given by the minimum cost required to "move" the probability mass from points in <img src="https://latex.codecogs.com/gif.latex?B" /> to the points in <img src="https://latex.codecogs.com/gif.latex?A" />. The OT distance is also known as the Earth Mover's Distance (EMD) as which can be viewed as the minimum amount of work required to transform one pile of earth into another. OT has several applications in machine learning.

Exact algorithms for computing OT can be slow, which has motivated the development of approximate numerical solvers (e.g. Sinkhorn method). We introduce a new and very simple combinatorial approach to find an <img src="https://latex.codecogs.com/gif.latex?\varepsilon" />-approximation of the OT distance. Our algorithm achieves a near-optimal execution time of <img src="https://latex.codecogs.com/gif.latex?O(n^2/\varepsilon^2)" /> for computing OT distance and, for the special case of the assignment problem, the execution time improves to <img src="https://latex.codecogs.com/gif.latex?O(n^2/\varepsilon)" />. Our algorithm is based on the push-relabel framework for min-cost flow problems.

So far, we have completed the implementation of the assignment algorithm. In this case, <img src="https://latex.codecogs.com/gif.latex?A" /> and <img src="https://latex.codecogs.com/gif.latex?B" /> each contain <img src="https://latex.codecogs.com/gif.latex?n" /> points and every point in <img src="https://latex.codecogs.com/gif.latex?A" /> (resp. <img src="https://latex.codecogs.com/gif.latex?B" />) has a demand of <img src="https://latex.codecogs.com/gif.latex?1/n" /> (resp. supply of <img src="https://latex.codecogs.com/gif.latex?1/n" />). We provide a CPU and a GPU implementation, and in the experiment, our algorithm is faster than the Sinkhorn algorithm both in CPU and GPU implementation. 

The OT implementation is a work in progress.

This repository contains two parts: 

1. Implementation of assignment algorithm: `matching.py`
2. Experiments compare our method and Sinkhorn: `pl_vs_sinkorn_mnist.py`,`pl_vs_sinkorn_synthetic.py`

### Dependencies

To use our algorithm or reproduce our experiments, simply install the following dependencies in your python environment and run the code.

For the first part, our algorithm implementation requires:

- [NumPy](https://numpy.org/install/) v1.21 
- [PyTorch](https://pytorch.org/) v1.10

Reproducing our experiments requires:

- [CUDA Toolkit](https://developer.nvidia.com/cuda-11.3.0-download-archive) v11.3
- [CuPy](https://docs.cupy.dev/en/stable/install.html) v9.6
- [NumPy](https://numpy.org/install/) v1.21
- [POT](https://pythonot.github.io/) v0.8.1
- [PyTorch](https://pytorch.org/) v1.10
- [TensorFlow](https://www.tensorflow.org/install) v2.6

We would like to thank [Abhijeet Phatak](https://github.com/abhijit-15) for sharing his knowledge about GPUs, which helped inform our GPU-based implementation.
## 
