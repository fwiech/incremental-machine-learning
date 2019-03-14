# Incremental Machine Learning
comparative evaluation of incremental machine learning methods

## Elastic Weight Consolidation
This is an implementation of "Overcoming catastrophic forgetting in neural networks" (https://arxiv.org/abs/1612.00796) for supervised learning in TensorFlow.

## Running Code

### Show Commands

`python3 main.py -h`

### Examples

| Example | First Task | Second Task |
| ------------- | ------------- | ------------- |
| 9 / 1 | `python3 main.py 0 1 2 3 4 5 6 7 8 0.001 20000 128 '' 1 -s 'a'` | `python3 main.py 9 0.00001 50 128 'a' 1200`
| 5 / 5 | `python3 main.py 0 1 2 3 4 0.001 20000 128 '' 1 -s 'a'` | `python3 main.py 5 6 7 8 9 0.00001 50 128 'a' 1200`
