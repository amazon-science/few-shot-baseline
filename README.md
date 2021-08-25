# A Baseline for Few-Shot Image Classification

This repository contains the code for the paper:

[Guneet Singh Dhillon](https://guneet-dhillon.github.io/), [Pratik Chaudhari](https://pratikac.github.io/), Avinash Ravichandran, [Stefano Soatto](http://web.cs.ucla.edu/~soatto/)  
**A Baseline for Few-Shot Image Classification** ([pdf](https://openreview.net/pdf?id=rylXBkrYDS))  
*In Proceedings of the International Conference on Learning Representations (ICLR), 2020*

## Abstract

Fine-tuning a deep network trained with the standard cross-entropy loss is a strong baseline for few-shot learning.
When fine-tuned transductively, this outperforms the current state-of-the-art on standard datasets such as Mini-ImageNet, Tiered-ImageNet, CIFAR-FS and FC-100 with the same hyper-parameters.
The simplicity of this approach enables us to demonstrate the first few-shot learning results on the ImageNet-21k dataset.
We find that using a large number of meta-training classes results in high few-shot accuracies even for a large number of few-shot classes.
We do not advocate our approach as the solution for few-shot learning, but simply use the results to highlight limitations of current benchmarks and few-shot protocols.
We perform extensive studies on benchmark datasets to propose a metric that quantifies the "hardness" of a few-shot episode.
This metric can be used to report the performance of few-shot algorithms in a more systematic way.

## Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{
Dhillon2020A,
title={A Baseline for Few-Shot Image Classification},
author={Guneet Singh Dhillon and Pratik Chaudhari and Avinash Ravichandran and Stefano Soatto},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=rylXBkrYDS}
}
```

## Usage

### Dataset

Each dataset has a folder of its own, with individual README files.

Meta-Dataset can be used with this code-base as well. See the paper for the complete set of results.

### Meta-Training

The meta-training phase takes the following arguments:
- dataset           : the dataset to use
- architecture      : the model architecture to use
- train and val     : whether to use both the train and validation sets

Examples:
- To meta-train on the train set of Mini-ImageNet using a WideResNet-28-10, run
```
python -m fastai.launch pretrain.py --dataset Mini-ImageNet --architecture WRN2810
```
- To meta-train on the train and validation sets of Mini-ImageNet using a WideResNet-28-10, run
```
python -m fastai.launch pretrain.py --dataset Mini-ImageNet --architecture WRN2810 --train_and_val True
```

### Few-Shot Training and Testing

After meta-testing, the few-shot training and testing phase takes the following arguments:
- dataset           : the dataset to use
- architecture      : the model architecture to use
- way               : the number of ways (few-shot classes) to use
- support shot      : the number of support shots (labeled train examples per class) to use
- query shot        : the number of query shots (unlabeled test examples per class) to use
- num episodes      : the number of few-shot episodes to use
- train and val     : whether to use the meta-trained model trained on both the train and validation sets
- non transductive  : whether to fine-tuning non-transductively

Examples:
- To few-shot train transductively and test on 1000 few-shot episodes from Mini-ImageNet with a way of 5, support shot of 1 and queryshot of 15, using a WideResNet-28-10 trained on the train set, run
```
python finetune.py --dataset Mini-ImageNet --architecture WRN2810 --way 5 --support_shot 1 --query_shot 15 --num_episodes 1000
```
- To few-shot train non-transductively and test on 1000 few-shot episodes from Mini-ImageNet with a way of 5, support shot of 1 and queryshot of 15, using a WideResNet-28-10 trained on the train set, run
```
python finetune.py --dataset Mini-ImageNet --architecture WRN2810 --way 5 --support_shot 1 --query_shot 15 --num_episodes 1000 --non_transductive
```

### Systematic Evaluation

After few-shot training and testing, the performance of the algorithm can be visualized against the hardness of each few-shot episode, per dataset. To do so, run
```
python plot.py
```
