# Python 2/3 compatiblity
import os
from utils import get_device
from models import get_model, test
import numpy as np
from skimage import io
import argparse
import torch
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
import time
from utils import metrics, print_results, timer, applyPCA,convert_to_color,setPalette,setMask
from functools import lru_cache
import ctypes
import sys



#gestione codice in C++
dir_path = os.path.dirname(os.path.realpath(__file__))
handle = ctypes.CDLL(dir_path + "/pca.so")     
handle.myFunction.argtypes = [ctypes.c_int, ctypes.c_int] 
  
def cudaPca(K,ds):
    return handle.myFunction(K,ds) 


# Test options
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
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

parser.add_argument(
    "--image",
    type=str,
    default=None,
    nargs="?",
    help="Path to an image on which to run inference.",
)

@lru_cache(maxsize=128, typed=False)
def main():
    args = parser.parse_args()
    CUDA_DEVICE = get_device(args.cuda)
    MODEL = "li"
    INFERENCE = args.image
    CHECKPOINT = args.checkpoint
    PCAnum = args.pca
	
    startTot = time.time()
    img_filename = os.path.basename(INFERENCE)
    basename = MODEL + img_filename 
    dirname = os.path.dirname(INFERENCE)

    img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(INFERENCE, "")
    # Normalization
    img = np.asarray(img, dtype="float32")
    #valori normalizzati tra 0 e 1
    #img = (img - np.min(img)) / (np.max(img) - np.min(img))
    N_CLASSES = len(LABEL_VALUES)
    
    if(INFERENCE[0] == "PaviaU"):
        ds=1
    else: 
        ds=2

    cudaPca(PCAnum,ds)
    #imgP,pca = applyPCA(img,gt,numComponents=PCAnum,flag=1)
    #imgP = (imgP - np.min(imgP)) / (np.max(imgP) - np.min(imgP))

    #load cuclas mat file
    imgC = open_file("CublasPCA.mat")["X"]
    # Normalization
    imgC = np.asarray(imgC, dtype="float32")
    #imgC = (imgC - np.min(imgC)) / (np.max(imgC) - np.min(imgC))
    

    N_BANDS = imgC.shape[-1]
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
    probabilities = test(model, imgC, hyperparams)
    prediction = np.argmax(probabilities, axis=-1)
    endT = time.time()
    print("Tempo Inference: ")
    timer(startT,endT)

    prediction = setMask(prediction,gt,IGNORED_LABELS)

    run_results = metrics(
            prediction,
            gt,
            ignored_labels=hyperparams["ignored_labels"],
            n_classes=N_CLASSES,
        )
	
    endTot = time.time()
    print("Tempo totale: ")
    timer(startTot,endTot)

    print_results(run_results,label_values=LABEL_VALUES)
    io.imsave("predictionCublas.tif", convert_to_color(prediction))
	

main()
