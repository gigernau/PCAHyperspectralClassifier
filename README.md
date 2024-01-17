# PCAHyperspectralClassifier
<img src="https://github.com/gigernau/PCAHyperspectralClassifier/blob/main/logo.png" align="left" height="250" width="300">
<br><br><br><br><br>
Classification of Hyperspectral Images ( HSIs ) with Principal Component Analysis ( PCA ) preprocessing exploiting CUDA ( cuBLAS ).
The code, explicitly designed for the NVIDIA Jetson Nano Developer kit, can run on any PC with NVIDIA GPU, Python3, and the necessary modules.
<br><br><br><br><br>

[![DOI](https://zenodo.org/badge/441206629.svg)](https://zenodo.org/doi/10.5281/zenodo.10522970)

## References

This work uses the [DeepHyperX](https://github.com/nshaud/DeepHyperX) toolbox based on the following paper in Geoscience and Remote Sensing Magazine :
> N. Audebert, B. Le Saux and S. Lefevre, "*Deep Learning for Classification of Hyperspectral Data: A Comparative Review*," in IEEE Geoscience and Remote Sensing Magazine, vol. 7, no. 2, pp. 159-173, June 2019.

For the PCA, this work uses the algorithm of this paper :
> M. Andrecut, "Parallel GPU Implementation of Iterative PCA Algorithms*," 2009, https://www.researchgate.net/publication/26829736_Parallel_GPU_Implementation_of_Iterative_PCA_Algorithms. 

as implemented on https://github.com/nmerrill67/GPU_GSPCA

As 3D CNN, this work uses the model described in this paper :
>   * 3D CNN ([Spectral–Spatial Classification of Hyperspectral Imagery with 3D Convolutional Neural Network, Li et al., Remote Sensing 2017](http://www.mdpi.com/2072-4292/9/1/67))



Bibtex format :

> @article{8738045,
author={N. {Audebert} and B. {Le Saux} and S. {Lefèvre}},
journal={IEEE Geoscience and Remote Sensing Magazine},
title={Deep Learning for Classification of Hyperspectral Data: A Comparative Review},
year={2019},
volume={7},
number={2},
pages={159-173},
doi={10.1109/MGRS.2019.2912563},
ISSN={2373-7468},
month={June},}

> @misc{pca, 
    author={M. Andrecut},
    title= {Parallel GPU Implementation ofIt is based on the PyTorch deep learning and GPU computing framework
Iterative PCA Algorithms}, 
    journal={Research Gate},
    year={2009},
     note ={\url{https://www.researchgate.net/publication/26829736_Parallel_GPU_Implementation_of_Iterative_PCA_Algorithms}}}
     
> @Article{rs9010067,
AUTHOR = {Li, Ying and Zhang, Haokui and Shen, Qiang},
TITLE = {Spectral–Spatial Classification of Hyperspectral Imagery with 3D Convolutional Neural Network},
JOURNAL = {Remote Sensing},
VOLUME = {9},
YEAR = {2017},
NUMBER = {1},
ARTICLE-NUMBER = {67},
URL = {https://www.mdpi.com/2072-4292/9/1/67},
ISSN = {2072-4292},
DOI = {10.3390/rs9010067}
}

     

## DATASET
Several public hyperspectral datasets are available at [UPV/EHU](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes) wiki.
For the testing of this code, we used the following public datasets:
```
PaviaU
    ├── PaviaU_gt.mat
    └── PaviaU.mat
    
IndianPines
    ├── IndianPines_gt.mat
    └── IndianPines_corrected.mat
    
Salinas
    ├── Salinas_gt.mat
    └── Salinas_corrected.mat
```

# SETUP FOR UBUNTU LINUX OS

## 1) Update : 
	sudo apt update

## 2) Install pip3 for Python3: 
	sudo apt install python3-pip  && python3 -m pip install --upgrade pip

## 3) Install Python3 modules : 
	python3 -m pip install -r requirements.txt
	
## 4) Check Cuda version or Install [CUDA](https://developer.nvidia.com/cuda-toolkit)
	nvcc -V

## 5) Install CuPy module ( e.g., for CUDA 12.1 )
	python3 -m pip install cupy-cuda12x

## 6) Install gsl library:
	sudo apt-get install libgsl-dev
  
## 7) Compile PCA with CUDA compiler
	nvcc -Xcompiler -fPIC -shared -o pca.so main.cpp kernel_pca.cu -lcublas -lm -lgsl -lgslcblas
	
# USAGE EXAMPLE

## 1) Set VISDOM environment in a separate shell to view images and plots in a browser
	python3 -m visdom.server
	
## 2) Train the Model
	python3 main.py --model li --dataset IndianPines --training_sample 0.7  --epoch 200 --cuda 0 --pca 10
	
## 3) Inference
	python3 inference.py --cuda 0 --image IndianPines --checkpoint models/ip/10_IP.pth --model li --pca 10


## To sample milliWatts absorbed every second on Jetson Nano, run on a separate terminal
	sudo watch -t -n 1 "(cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power1_input) | tee -a consumpPCA.txt"



[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/gianluca.delucia)

