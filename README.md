## Gaussian RAM: Lightweight Image Classification via Stochastic Retina Inspired Glimpse and Reinforcement Learning

### ICROS ICCAS 2020 Accepted

Official PyTorch implementation of Gaussian-RAM


## Abstract
Previous studies on image classification have been mainly focused on the performance of the networks, not onreal-time operation or model compression.  We propose a Gaussian Deep Recurrent visual Attention Model (GDRAM)- a reinforcement learning based lightweight deep neural network for large scale image classification that outperformsthe conventional CNN (Convolutional Neural Network) which uses the entire image as input.  Highly inspired by thebiological visual recognition process, our model mimics the stochastic location of the retina with Gaussian distribution. We evaluate the model on Large cluttered MNIST, Large CIFAR-10 and Large CIFAR-100 datasets which are resized to128 in both width and height.

<p align = "center">
<img src="https://github.com/dsshim0125/gaussian-ram/blob/master/fig.png" width="600"> 
</p>

## Dataset
Cluttered MNIST([download](https://drive.google.com/file/d/1nMO5XIFmjyPnJjfvBeFpujeuZ3Qk7vhd/view?usp=sharing)), CIFAR10 and CIFAR100 are used to train and evaluate. All the images are resized to 128 in both height and weight for generating high scale image.
## Requirements
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)
- PIL
- NumPy

## Training
```angular2html
python train.py --data_path --dataset --batch_size --lr --epochs --random_seed --log_interval --resume --checkpoint
```

## Inference
```angular2html
python inference.py --data_path --dataset --random_seed --fast
```

## Acknowledgement
This work was supported by Institute of Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government (MSIT) (No. 2019-0-01367, Infant-Mimic Neurocognitive Developmental Machine Learning from Interaction Experience with Real World (BabyMind))
