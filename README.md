# Dynamic Inertial Poser (DynaIP): Part-Based Motion Dynamics Learning for Enhanced Human Pose Estimation with Sparse Inertial Sensors

This is the pytorch implementation of our paper DynaIP at CVPR 2024.

Arxiv: https://arxiv.org/abs/2312.02196

## Environment Setup

We tested our code on Windows with `Python 3.8.15`, `Pytorch 1.10.2` with `cuda11.1`, other dependencies are specified in `requirements.txt`.

```python
conda create -n dynaip python==3.8
conda activate dynaip
pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 torchaudio==0.10.2 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```

## Datasets and Models

### Datasets

We used publicly available Xsens Mocap datasets and DIP-IMU to train and evaluate our model. These Xsens datasets include AnDy, UNIPD-BPE, Emokine, CIP and Virginia Natural Motion. You can download the raw data from:

+ AnDy: https://zenodo.org/records/3254403 (xsens_mvnx.zip)
+ UNIPD: https://doi.org/10.17605/OSF.IO/YJ9Q4  (we use all the .mvnx files in single_person folder)
+ Emokine: https://zenodo.org/records/7821844
+ CIP: https://doi.org/10.5281/zenodo.5801928  (MTwAwinda.zip)
+ Virginia Natural Motion: https://doi.org/10.7294/2v3w-sb92
+ DIP-IMU: https://dip.is.tue.mpg.de

Clone this repo, download the above datasets, extract and place them in `./datasets/raw/`.

```python
datasets
├─extract
├─raw
│  ├─andy
│  ├─cip
│  ├─dip
│  ├─emokine
│  ├─unipd
│  └─virginia
└─work
```

### SMPL Models

We used the smpl model, download it from [here](https://smpl.is.tue.mpg.de) and place in  `./smpl_models/`.

```python
smpl_models
    smpl_female.pkl
    smpl_male.pkl
    smpl_neutral.pkl
```

## Data Processing

1. Run `extract.py` , this will extract imu and pose data from raw .mvnx files, downsampling them to 60Hz. After data extraction, you can use scripts in `extract.py` for visualization. 

```python
python ./datasets/extract.py
```

> Note that Virginia Natural Motion has pose drifts due to long time tracking, we visualized part of its sequence and manually selected clean frames as training and evaluation data, those selected frames are also stored in `extract.py`.

2. Run `process.py` to preprocess IMU data from extracted Xsens datasets and raw DIP-IMU. 

```python
python ./datasets/process.py
```

> Since DIP-IMU has no root trajectory, we generate pseudo-root trajectory by forcing the lowest foot to touch the ground. The training and test split information of each dataset is stored in `./datasets/split_info/`. 

## Training and Evaluation

For training and evaluation, simply run:

```python
python train.py
python evaluation.py
```

The pretrained weights are stored in `./weights` folder.

## Visualization

We use `aitviewer` for visualization, run:

```python
python vis.py
```

for visualizing the predicted results.

## Acknowledgement

Some of our codes are adapted from [PIP](https://github.com/Xinyu-Yi/PIP) and [VT-Natural-Motion-Processing](https://github.com/ARLab-VT/VT-Natural-Motion-Processing).

## Citation

If you find this project helpful, please consider citing us:

```
@article{zhang2023dynamic,
  title={Dynamic Inertial Poser (DynaIP): Part-Based Motion Dynamics Learning for Enhanced Human Pose Estimation with Sparse Inertial Sensors},
  author={Zhang, Yu and Xia, Songpengcheng and Chu, Lei and Yang, Jiarui and Wu, Qi and Pei, Ling},
  journal={ IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2024},
  publisher={IEEE},
  booktitle={cvpr}
}
```

