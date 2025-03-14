# MASA
Source code for for our project: MASA: Multi-view Adaptive Subspace Alignment for Enhanced
Few-Shot Learning

## Environments
The proposed RAG is implemented with python 3.11 on GPU. 
All results in the paper are from running on NVIDIA RTX3080Ti GPU.

### Main Packages
+ Python: 3.11.10
+ PyTorch: 2.5.1
+ CUDA: 12.4
+ Torchvision: 0.20.1
+ Torchaudio: 2.5.1
+ Torchtriton: 3.1.0
+ NumPy: 1.26.4
+ SciPy: 1.13.1
+ Pillow: 10.4.0

If you are using Anaconda, an identical environment can also be created by using the following command:
```conda env create -f environment.yml```

## Datasets
The 4 datasets we used: MV_grass, MV_paper, NUS_WIDE_OBJECT, Caltech-20.
Here we show the experiments on two of these datasets.

The ```data_process/``` includes preprocessing of MV_grass and NUS_WIDE_OBJECT data.

The ```MV_grass_dataset/``` includes the original dataset, which is the RGB image.

## Quick Start
All running files are in ```run/```.

Running classification experiments on the MV_grass dataset(3 view): ```python MV_grass.py```. Just select the parameter of C and K of tasks you need.

Running classification experiments on the NUS__WIDE_OBJECT(5 view): ```python NUS_5view.py```. Just select the parameter of C and K of tasks you need.

(Before running, please make sure that the data path has been changed to the path you saved.)
