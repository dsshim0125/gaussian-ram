## Submission to IEEE ICCAS 2020
## Gaussian RAM: Lightweight Image Classification via Stochastic Retina Inspired Glimpse and Reinforcement Learning


Official PyTorch implementation of Gaussian-RAM


## Introduction
Previous studies on image classification have been mainly focused on the performance of the networks, not onreal-time operation or model compression.  We propose a Gaussian Deep Recurrent visual Attention Model (GDRAM)- a reinforcement learning based lightweight deep neural network for large scale image classification that outperformsthe conventional CNN (Convolutional Neural Network) which uses the entire image as input.  Highly inspired by thebiological visual recognition process, our model mimics the stochastic location of the retina with Gaussian distribution.We evaluate the model on Large cluttered MNIST, Large CIFAR-10 and Large CIFAR-100 datasets which are resized to128 in both width and height.

<img src="https://github.com/dsshim0125/gaussian-ram/blob/master/img/fig.png" width="400"> 
<p align = "center">


### Requirements
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)
- NumPy
