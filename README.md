# Classification-of-Hyperspectral-Image-HSI-with-Principal-Component-Analysis-PCA-in-CUDA-cuBLAS-

Classification of Hyperspectral Image HSI with Principal Component Analysis PCA in CUDA ( cuBLAS ) Master Degree Thesis in Computer Science.

This part of work contains only inferece part (whithout train/test model script).


## Reference

This work was used [DeepHyperX](https://github.com/nshaud/DeepHyperX) toolbox based on this paper in Geoscience and Remote Sensing Magazine :
> N. Audebert, B. Le Saux and S. Lefevre, "*Deep Learning for Classification of Hyperspectral Data: A Comparative Review*," in IEEE Geoscience and Remote Sensing Magazine, vol. 7, no. 2, pp. 159-173, June 2019.

For the PCA this work is based on this inplementation of this paper :
> M. Andrecut, "Parallel GPU Implementation of Iterative PCA Algorithms*," 2009, https://www.researchgate.net/publication/26829736_Parallel_GPU_Implementation_of_Iterative_PCA_Algorithms. 

For the use of 3D CNN this work is based on this paper :
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
     

## DATASET
Several public hyperspectral datasets are available on the [UPV/EHU](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes) wiki.
This work use the following public datasets:
```
PaviaU
    ├── PaviaU_gt.mat
    └── PaviaU.mat
```

# SETUP FOR LINUX OS
## FOR THE FIRT USE:

0) update everything: 
> sudo apt update && sudo apt upgrade

1) install python3 : 
> sudo apt-get install python3.7

2) install pip3 : 
> sudo apt install python3-pip  && python3 -m pip install --upgrade pip

3) install python modules : 
> python3 -m pip install -r requirements.txt

4) install Pythorch:
	> sudo apt install libopenblas-base libopenmpi-dev
	

### FOR C++ CUDA, CUBLAS and MATIO

5) install gls libraries e cblas:
	> sudo apt-get install libgsl-dev
  
6) install Cuda:
	
>	echo "# Add CUDA bin & library paths:" >> ~/.bashrc
	echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc
	echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
	source ~/.bashrc

7) install matio-cpp  (https://github.com/ami-iit/matio-cpp):
> 	sudo apt install libmatio-dev
	git clone https://github.com/dic-iit/matio-cpp
	cd matio-cpp
	mkdir build && cd build
	cmake ../
	make
	[sudo] make install



# EXAMPLE OF USE

1) Compile and share PCA library in C++ Cuda
> nvcc -Xcompiler -fPIC -shared -o pca.so main.cpp kernel_pca.cu -lcublas -lm -lgsl -lgslcblas -lmatioCpp

2) INFERCENCE
> python3 inference.py --pca 10 --image PaviaU --cuda 0 --checkpoint model.pth


## TO KNOW ENERGY CONSUPTION in milliWatt/sec
	 > sudo watch -t -n 1 "(cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power1_input) | tee -a consumiPCA.txt"

[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/nshaud)
