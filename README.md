# Differential-Privacy-CIFAR-dataset

This repository contains the code of training deep learning model on a noisy CIFAR dataset. The noisy dataset is generated after adding a Gaussian noise to each image. In what follows, we describe each file of this repository:

mainCifarNormalNoise.py: This file contains the main function for:
                           1) pre-processing the CIFAR dataset, i.e., normalizing the dataset and adding Normal noise
                           2) training the learning model located in the file cifarNet.py

cifarNet.py            : This file contains the designed learning model
TransformNormal.py     : This file contains the transform for adding the Normal noise to the original dataset before starting the training


 
