# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division
import joblib
import os
from utils import convert_to_color_, convert_from_color_, get_device
from datasets import open_file
from models import get_model, test
import numpy as np

from skimage import io
import argparse
import torch

from torch import Tensor, nn

from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
import time
from utils import metrics, print_results, timer,convert_to_color,setPalette,setMask
from functools import lru_cache
import ctypes
import sys
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
import scipy
import subprocess
import cupy as cp

#gestione codice in C++
dir_path = os.path.dirname(os.path.realpath(__file__))
handle = ctypes.CDLL(dir_path + "/pca.so")   
handle.cudaPCA.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
                            ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                            #np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')]
                            ctypes.POINTER(ctypes.c_float)]   
handle.cudaPCA.restype = ctypes.c_void_p
  
def cudaPCA(img,K,d0,d1,d2,imgT):
    return handle.cudaPCA(img,K,d0,d1,d2,imgT)



# Test options
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Model to train. Available:\n"
    "li (3D CNN), "
)
parser.add_argument(
    "--cuda",
    type=int,
    default=-1,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Weights to use for initialization, e.g. a checkpoint",
)

parser.add_argument(
        "--pca",
        type=int,
        default=-1,
        help="Specify PCA numeber component (defaults to -1, no PCA)",)

group_test = parser.add_argument_group("Test")

group_test.add_argument(
    "--image",
    type=str,
    default=None,
    nargs="?",
    help="Path to an image on which to run inference.",
)


# Training options
group_train = parser.add_argument_group("Model")
group_train.add_argument(
    "--patch_size",
    type=int,
    help="Size of the spatial neighbourhood (optional, if "
    "absent will be set by the model)",
)
group_train.add_argument(
    "--batch_size",
    type=int,
    help="Batch size (optional, if absent will be set by the model",
)


@lru_cache(maxsize=128, typed=False)
def main():
    args = parser.parse_args()
    CUDA_DEVICE = get_device(args.cuda)
    MODEL = args.model
    INFERENCE = args.image
    CHECKPOINT = args.checkpoint
    PCAnum = args.pca

    print("Dataset: " + str(INFERENCE))
    print("PCA: " + str(PCAnum))
    
    img_filename = os.path.basename(INFERENCE)
    basename = MODEL + img_filename 
    dirname = os.path.dirname(INFERENCE)

    img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(INFERENCE,target_folder="./Datasets/")

    N_CLASSES = len(LABEL_VALUES)


    if(PCAnum>=1):
        
        imgT = cp.zeros(img.shape[0]*img.shape[1]*PCAnum, dtype=cp.float32)
        imgTx = ctypes.cast(imgT.data.ptr, ctypes.POINTER(ctypes.c_float))

        im = img.flatten(order='F')
        startT = time.time()
        cudaPCA(im,PCAnum,img.shape[0],img.shape[1],img.shape[2],imgTx)
        endT = time.time()
        print(f"PCA Time:")
        timer(startT,endT)
        # Normalization
        img = cp.reshape(imgT.astype(cp.float32),[img.shape[0],img.shape[1],PCAnum],order='F')
    
    



    startTot = time.time()

    N_BANDS = img.shape[-1]
    hyperparams  = vars(args)
    hyperparams.update(
        {
            "n_classes": N_CLASSES,
            "n_bands": N_BANDS,
            "device": CUDA_DEVICE,
            "ignored_labels": [0],
            "test_stride": 1,
        }
    )
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    palette = setPalette(N_CLASSES)
    model, _, _, hyperparams = get_model(MODEL, **hyperparams)
    model.load_state_dict(torch.load(CHECKPOINT))

    startT = time.time()
    probabilities = test(model, img, hyperparams)
    prediction = np.argmax(probabilities, axis=-1)
    
    endT = time.time()
    print("Inference Time: ")
    timer(startT,endT)
	
    endTot = time.time()
    print("Total Time: ")
    timer(startTot,endTot)

    io.imsave("prediction_inference.tif", convert_to_color(prediction))


main()