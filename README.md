## GAS-Net

The pytorch implementation for "[Global-Aware Siamese Network for Change Detection on Remote Sensing Images]" on ISPRS JOURNAL OF PHOTOGRAMMETRY AND REMOTE SENSING. 

The **GAS-Net** is designed to generate global-aware features for efficient change detection by incorporating the relationships between scenes and foregrounds.


## Results

![image](https://raw.githubusercontent.com/xiaoxiangAQ/GAS-Net/main/doc/result1.jpg)

![image](https://github.com/xiaoxiangAQ/GAS-Net/tree/main/doc/result2.jpg)


## Requirements

- Python 3.6
- Pytorch 1.2.0


## Datasets

- Download the [Levir-CD Dataset](https://justchenhao.github.io/LEVIR/)
- Download the [Lebediv-CD Dataset](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-2/565/2018/isprs-archives-XLII-2-565-2018.pdf)


The data folder is structured as follows:

```
├── data/
│   ├── levir_CD/    # Levir-CD dataset
|   |   ├── train/    # traning set 
|   |   |   ├── t1/    #images of time t1
|   |   |   ├── t2/    #images of time t2
|   |   |   ├── label/    #ground truth
|   |   ├── val/    # validation set
|   |   |   ├── t1/
|   |   |   ├── t2/
|   |   |   ├── label/
|   |   ├── test/    # testing set
|   |   |   ├── t1/
|   |   |   ├── t2/
|   |   |   ├── label/    #ground truth for evaluation
|   |   ├── results/    # path to save the model
│   ├── SVCD/
|   |   ├── leveb/    # Lebediv-CD dataset, have the same structure of the Levir-CD dataset
...
```


## Citation

@article{Global2023zhang,<br>
title = {Global-aware siamese network for change detection on remote sensing images},<br>
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},<br>
volume = {199},<br>
pages = {61-72},<br>
year = {2023},<br>
doi = {https://doi.org/10.1016/j.isprsjprs.2023.04.001},<br>
author = {Ruiqian Zhang and Hanchao Zhang and Xiaogang Ning and Xiao Huang and Jiaming Wang and Wei Cui},<br>
}


## Acknowledgment

This code is heavily borrowed from [SRCDNet](https://github.com/leftthomas/SRGAN).
