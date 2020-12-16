# Features Extraction and Transfer Learning

MATH6380P Project 1, Fall 2020

Hong Wing PANG, Yik Ben WONG

## Introduction

In this repository we provide the code for the following tasks:

- Feature extraction on Scattering Net [[1]](#1) (to be uploaded soon)
- Re-training of VGG11 and ResNet18 models over `torchvision` pre-trained weights
- Classification algorithms on extracted features
- Evaluation of *neural collapse* statistics proposed in [[2]](#2).

Our findings are covered in detail in our [project report](https://github.com/ferrophile/fentl/blob/master/report.pdf).

## Environment

The code is tested on PyTorch 1.5.1 and Python 3.7. Training is mostly done on the 
ITSC HPC2 cluser with NVIDIA Tesla K80 x2.

See `requirements.txt` for a list of installed packages.

## Getting started

### Training

Create a directory `./checkpoints` for storing the weights. Then run the following
to train for 10 epochs:

```
python3 train.py --batch_size 512, --gpu_ids 0,1,2,3 --model <MODEL>, --dataset <DATASET>,
--num_workers 0, --niters 10
```
where `model` is `vgg11` / `resnet18` and `dataset` is `mnist` / `fashion`.

Customize the batch size, no. of GPUs and no. of CPU threads (for data loading)
based on amount of resources available on your machine.

### Evaluation

Extract last-layer activations and run evaluation as follows:

```
python3 eval.py --batch_size 512, --gpu_ids 0,1,2,3 --model <MODEL>, --dataset <DATASET>,
--num_workers 0, --classifiers log_reg,svm,lda,rand_forest --which_epoch 01
```

Specify `--which_epoch` to evalute the weights of some epoch saved during training.
If not specified, the pre-trained model will be evaluated.

Specify `--classifiers` as a comma-separated list of classfiers you wish to
evaluate. The list of available classifiers are listed above. If not specified, no
classification will be performed on the feature vectors.

## References

<a id="1">[1]</a>
Mallat, Stéphane. "Group invariant Scattering." Communications on Pure and Applied Mathematics 65.10
(2012): 1331-1398.

<a id="2">[2]</a> V. Papyan, X. Y. Han,; D.L Donoho. Prevalence of neural collapse during the terminal
phase of deep learning training. Proceedings of the National Academy of Sciences, 202015509.