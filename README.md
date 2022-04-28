# TransBTS: Multimodal Brain Tumor Segmentation Using Transformer（Accepted by MICCAI2021）
# TransBTSV2: Towards Better and More Efficient Volumetric Segmentation of Medical Images

This repo is the official implementation for: 
1) [TransBTS: Multimodal Brain Tumor Segmentation Using Transformer](https://arxiv.org/pdf/2103.04430.pdf). 
2) [TransBTSV2: Towards Better and More Efficient Volumetric Segmentation of Medical Images](https://arxiv.org/abs/2201.12785). 

## Requirements
1)The multimodal brain tumor datasets (BraTS 2019 & BraTS 2020) could be acquired from [here](https://ipp.cbica.upenn.edu/).
2)The liver tumor dataset LiTS 2017 could be acquired from [here](https://competitions.codalab.org/competitions/17094#participate-get-data).
3)The kidney tumor dataset KiTS 2019 could be acquired from [here](https://kits19.grand-challenge.org/data/).

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

Quantitive comparison of performance on BraTS2019 validation set as well as BraTS2020 validation set between our proposed TransBTS with other SOTA methods.

![quantitive_comparison](https://github.com/Wenxuan-1119/TransBTS/blob/main/figure/quantitive_comparison.PNG "quantitive_comparison")

## Visual comparison
Here are some samples from BraTS 2019 dataset for visual comparison between our proposed TransBTS with other SOTA methods.

![visual_comparison](https://github.com/Wenxuan-1119/TransBTS/blob/main/figure/visual_comparison.PNG "visual_comparison")

## Citation
If you use our code or models in your work or find it is helpful, please cite the corresponding paper:

```
@inproceedings{wang2021transbts,
  title={TransBTS: Multimodal Brain Tumor Segmentation Using Transformer},  
  author={Wang, Wenxuan and Chen, Chen and Ding, Meng and Li, Jiangyun and Yu, Hong and Zha, Sen},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2021}
}
```

```
@article{li2022transbtsv2,
  title={TransBTSV2: Wider Instead of Deeper Transformer for Medical Image Segmentation},
  author={Li, Jiangyun and Wang, Wenxuan and Chen, Chen and Zhang, Tianxiang and Zha, Sen and Yu, Hong and Wang, Jing},
  journal={arXiv preprint arXiv:2201.12785},
  year={2022}
}
```

## Reference
1.[setr-pytorch](https://github.com/gupta-abhay/setr-pytorch)

2.[BraTS2017](https://github.com/MIC-DKFZ/BraTS2017)


