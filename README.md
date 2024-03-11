# TransBTS（MICCAI2021）& TransBTSV2 (To Be Updated)

This repo is the official implementation for: 
1) [TransBTS: Multimodal Brain Tumor Segmentation Using Transformer](https://arxiv.org/abs/2103.04430). 

2) [TransBTSV2: Towards Better and More Efficient Volumetric Segmentation of Medical Images](https://arxiv.org/abs/2201.12785). 

The details of the our TransBTS and TransBTSV2 can be found at the models directory ([TransBTS](https://github.com/Wenxuan-1119/TransBTS/tree/main/models/TransBTS) and [TransBTSV2](https://github.com/Wenxuan-1119/TransBTS/tree/main/models/TransBTSV2)) in this repo or in the original paper.

## Requirements
- python 3.7
- pytorch 1.6.0
- torchvision 0.7.0
- pickle
- nibabel

## Data Acquisition
- The multimodal brain tumor datasets (**BraTS 2019** & **BraTS 2020**) could be acquired from [here](https://ipp.cbica.upenn.edu/).

- The liver tumor dataset **LiTS 2017** could be acquired from [here](https://competitions.codalab.org/competitions/17094#participate-get-data).

- The kidney tumor dataset **KiTS 2019** could be acquired from [here](https://kits19.grand-challenge.org/data/).

## Data Preprocess (BraTS 2019 & BraTS 2020)
After downloading the dataset from [here](https://ipp.cbica.upenn.edu/), data preprocessing is needed which is to convert the .nii files as .pkl files and realize data normalization.

`python3 preprocess.py`

## Training
Run the training script on BraTS dataset. Distributed training is available for training the proposed TransBTS, where --nproc_per_node decides the numer of gpus and --master_port implys the port number.

`python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train.py`

## Testing 
If  you want to test the model which has been trained on the BraTS dataset, run the testing script as following.

`python3 test.py`

After the testing process stops, you can upload the submission file to [here](https://ipp.cbica.upenn.edu/) for the final Dice_scores.

## Citation
If you use our code or models in your work or find it is helpful, please cite the corresponding paper:

- **TransBTS**:
```
@inproceedings{wang2021transbts,
  title={TransBTS: Multimodal Brain Tumor Segmentation Using Transformer},
  author={Wang, Wenxuan and Chen, Chen and Ding, Meng and Yu, Hong and Zha, Sen and Li, Jiangyun},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2021: 24th International Conference, Strasbourg, France, September 27--October 1, 2021, Proceedings, Part I 24},
  pages={109--119},
  year={2021},
  organization={Springer}
}
```

- **TransBTSV2**:
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


