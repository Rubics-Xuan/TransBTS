# TransBTS: Multimodal Brain Tumor Segmentation Using Transformer
This repo is the official implementation for TransBTS: Multimodal Brain Tumor Segmentation Using Transformer. The multimodal brain tumor dataset (BraTS 2019 and BraTS 2020) could be acquired from [here](https://ipp.cbica.upenn.edu/).

## TransBTS
![TransBTS](https://github.com/Wenxuan-1119/TransBTS/blob/main/figure/TransBTS.PNG "TransBTS")
Architecture of 3D TransBTS.

## Requirements
- pytorch 1.6.0
- python 3.7
- torchvision 0.7.0
- pickle
- nibabel

## Data preprocess
After downloading the dataset from [here](https://ipp.cbica.upenn.edu/), data preprocessing is needed which is to convert the .nii files as .pkl files and realize date normalization.
'python preprocess.py'
