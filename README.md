# TransBTS: Multimodal Brain Tumor Segmentation Using Transformer
This repo is the official implementation for [TransBTS: Multimodal Brain Tumor Segmentation Using Transformer](https://arxiv.org/pdf/2103.04430.pdf). The multimodal brain tumor dataset (BraTS 2019) could be acquired from [here](https://ipp.cbica.upenn.edu/).

## TransBTS
![TransBTS](https://github.com/Wenxuan-1119/TransBTS/blob/main/figure/TransBTS.PNG "TransBTS")
Architecture of 3D TransBTS.

## Requirements
- python 3.7
- pytorch 1.6.0
- torchvision 0.7.0
- pickle
- nibabel

## Data preprocess
After downloading the dataset from [here](https://ipp.cbica.upenn.edu/), data preprocessing is needed which is to convert the .nii files as .pkl files and realize date normalization.

`python3 preprocess.py`

## Training
Run the training script on BraTS dataset. Distributed training is available for training the proposed TransBTS, where --nproc_per_node decides the numer of gpus and --master_port implys the port number.

`python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train.py`

## Testing 
If  you want to test the model which has been trained on the BraTS dataset, run the testing script as following.

`python3 test.py`

After the testing process stops, you can upload the submission file to [here](https://ipp.cbica.upenn.edu/) for the final Dice_scores.

## Quantitive comparison of performance

Quantitive comparison of performance on BraTS2019 validation set between our proposed TransBTS with other SOTA methods.

![quantitive_comparison](https://github.com/Wenxuan-1119/TransBTS/blob/main/figure/quantitive_comparison.PNG "quantitive_comparison")

## Visual comparison
Here are some samples from BraTS 2019 dataset for visual comparison between our proposed TransBTS with other SOTA methods.

![visual_comparison](https://github.com/Wenxuan-1119/TransBTS/blob/main/figure/visual_comparison.PNG "visual_comparison")

## Citation

@article{wang2021transbts,
  title={TransBTS: Multimodal Brain Tumor Segmentation Using Transformer},  
  author={Wang, Wenxuan and Chen, Chen and Ding, Meng and Li, Jiangyun and Yu, Hong and Zha, Sen},
  journal={arXiv preprint arXiv:2103.04430},
  year={2021}
}

## Reference
[setr-pytorch](https://github.com/gupta-abhay/setr-pytorch)

