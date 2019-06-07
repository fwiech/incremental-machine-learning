# Comparative Evaluation of Incremental Machine Learning

## Abstract

The ability to learn and remember many different tasks is crucial for artificial intelligence. Neural networks are not capable of this. They suffer from catastrophic forgetting. Prior research in the domain of incremental or continual learning shows different approaches, such as [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796) or [Incremental Moment Matching](https://arxiv.org/abs/1703.08475). Some approaches rely on the computation of a so called Fisher information matrix. The Fisher information matrix shows rather promising results, but relies on a diagonal assumption and the context of Bayesian neural networks. Furthermore, the implementation of the Fisher information matrix in the machine learning framework Tensorflow requires a workaround that greatly increases memory consumtion.
This article proposes a new way of calculating a matrix that replaces the Fisher information matrix. It is computed similar as the Fisher information matrix, but does not requirde additional assumptions. Moreover, this matrix enables an easy computation in the Tensorflow framework. The article documents several benchmarks of an own reimplementation of the [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796) algorithm and the adoption of the new matrix.

## Quick start

Build all experiments and models with make:

```bash
make
```

### Benchmarks

#### D9-1

```bash
make D91
```

#### D5-5

```bash
make D55
```

#### P10-10

```bash
make PM
```
