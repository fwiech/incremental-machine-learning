# Incremental Machine Learning
comparative evaluation of incremental machine learning methods

## Elastic Weight Consolidation
This is an implementation of "Overcoming catastrophic forgetting in neural networks" (https://arxiv.org/abs/1612.00796) for supervised learning in TensorFlow.

## Running Code

### Show Help

`python3 main.py -h`

### Example

#### Task 0-8 & 9

```bash
python3 src/main.py --classes 0 1 2 3 4 5 6 7 8 --learnrate 0.001 --iterations 2500 --batch 100 --batch_fisher 1 --save FISH09
```
```bash
python3 src/main.py --classes 9 --learnrate 0.00001 --iterations 2500 --batch 100 --previous FISH09
```
