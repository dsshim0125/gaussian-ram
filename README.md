# Gaussian RAM

### ICROS ICCAS 2020 Student Best Paper Finalist

This repo is an official PyTorch implementation of "Gaussian RAM: Lightweight Image Classification via Stochastic Retina Inspired Glimpse and Reinforcement Learning". [[paper](https://arxiv.org/abs/2011.06190)]


## Abstract
Previous studies on image classification have been mainly focused on the performance of the networks, not on real-time operation or model compression.  We propose a Gaussian Deep Recurrent visual Attention Model (GDRAM)- a reinforcement learning based lightweight deep neural network for large scale image classification that outperformsthe conventional CNN (Convolutional Neural Network) which uses the entire image as input.  Highly inspired by the biological visual recognition process, our model mimics the stochastic location of the retina with Gaussian distribution. We evaluate the model on Large cluttered MNIST, Large CIFAR-10 and Large CIFAR-100 datasets which are resized to 128 in both width and height.

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
```bash
python train.py --data_path --dataset --batch_size --lr --epochs --random_seed --log_interval --resume --checkpoint
```

## Inference
```bash
python inference.py --data_path --dataset --random_seed --fast
```

## Acknowledgement
This work was supported by Institute of Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government (MSIT) (No. 2019-0-01367, Infant-Mimic Neurocognitive Developmental Machine Learning from Interaction Experience with Real World (BabyMind))

## References
[1]    Y.  Lecun,  L.  Bottou,  Y.  Bengio,  and  P.  Haffner,“Gradient-based   learning   applied   to   documentrecognition,” inProceedings of the IEEE, 1998, pp.2278–2324.<br />
[2]    K.  Simonyan  and  A.  Zisserman,  “Very  deep  con-volutional networks for large-scale image recogni-tion,”arXiv preprint arXiv:1409.1556, 2014.<br />
[3]    C. Szegedy,  W. Liu,  Y. Jia,  P. Sermanet,  S. Reed,D. Anguelov, D. Erhan, V. Vanhoucke, and A. Ra-binovich, “Going deeper with convolutions,” inPro-ceedings of the IEEE conference on computer visionand pattern recognition, 2015, pp. 1–9.<br />
[4]    K. He, X. Zhang, S. Ren, and J. Sun, “Deep resid-ual learning for image recognition,” inProceedingsof the IEEE conference on computer vision and pat-tern recognition, 2016, pp. 770–778.<br />
[5]    G. Huang,  Z. Liu,  L. Van Der Maaten,  and K. Q.Weinberger, “Densely connected convolutional net-works,” inProceedings of the IEEE conference oncomputer vision and pattern recognition, 2017, pp.4700–4708.<br />
[6]    Y. LeCun, “The mnist database of handwritten dig-its,”http://yann. lecun. com/exdb/mnist/.<br />
[7]    O.   Russakovsky,   J.   Deng,   H.   Su,   J.   Krause,S.   Satheesh,   S.   Ma,   Z.   Huang,   A.   Karpathy,A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei,“ImageNet  Large  Scale  Visual  Recognition  Chal-lenge,”International Journal of Computer Vision(IJCV), vol. 115, no. 3, pp. 211–252, 2015.<br />
[8]    V.  Mnih,  N.  Heess,  A.  Graveset al.,  “Recurrentmodels of visual attention,” inAdvances in neuralinformation processing systems,  2014,  pp.  2204–2212.<br />
[9]    J.  Ba,   V.  Mnih,   and  K.  Kavukcuoglu,   “Multi-ple object recognition with visual attention,”arXivpreprint arXiv:1412.7755, 2014.<br />
[10]  Q.  Liu,  R.  Hang,  H.  Song,  and  Z.  Li,  “Learn-ing  multi-scale  deep  features  for  high-resolutionsatellite    image    classification,”arXiv  preprintarXiv:1611.03591, 2016.<br />
[11]  M. Iftenea, Q. Liub, and Y. Wangc, “Very high res-olution images classification by fusing deep convo-lutional neural networks.”<br />
[12]  A.  Ablavatski,  S.  Lu,  and  J.  Cai,  “Enriched  deeprecurrent visual attention model for multiple objectrecognition,”  in2017 IEEE Winter Conference onApplications of Computer Vision (WACV).IEEE,2017, pp. 971–978.<br />
[13]  M.  Jaderberg,  K.  Simonyan,  A.  Zissermanet al.,
“Spatial  transformer  networks,”   inAdvances inneural information processing systems,  2015,  pp.2017–2025.<br />
[14]  J.   Redmon   and   A.   Farhadi,“Yolov3:Anincrementalimprovement,”arXiv   preprintarXiv:1804.02767, 2018.<br />
[15]  J. Choi, D. Chun, H. Kim, and H.-J. Lee, “Gaussianyolov3:  An accurate and fast object detector usinglocalization uncertainty for autonomous driving,” inProceedings of the IEEE International Conferenceon Computer Vision, 2019, pp. 502–511.<br />
[16]  S.   Ioffe   and   C.   Szegedy,    “Batch   normaliza-tion:   Accelerating  deep  network  training  by  re-ducing   internal   covariate   shift,”arXiv preprintarXiv:1502.03167, 2015.<br />
[17]  S. Hochreiter and J. Schmidhuber, “Long short-termmemory,”Neural computation,  vol.  9,  no.  8,  pp.1735–1780, 1997.<br />
[18]  R.  S.  Sutton,  D.  A.  McAllester,  S.  P.  Singh,  andY. Mansour, “Policy gradient methods for reinforce-ment learning with function approximation,” inAd-vances in neural information processing systems,2000, pp. 1057–1063.
