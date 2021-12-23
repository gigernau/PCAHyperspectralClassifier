# -*- coding: utf-8 -*-
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
from skimage import io

# Visualization
import seaborn as sns
import visdom

import os
from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    display_dataset,
    display_predictions,
    explore_spectrums,
    plot_spectrums,
    sample_gt,
    build_dataset,
    show_results,
    compute_imf_weights,
    get_device,
)
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model

import argparse

#PCA
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import chart_studio.plotly as py
import matplotlib.pyplot as plt

import time
import datetime


def showPCA(img,dt,gt, nbands):
    print("Fase Show PCA...")

    sns.axes_style('whitegrid')
    fig = plt.figure(figsize = (12, 6))

    for i in range(1, 1+6):
        fig.add_subplot(2,3, i)
        q = np.random.randint(img.shape[2])
        cls = px.imshow(img[:,:,q], color_continuous_scale='jet',)
        cls.update_layout(title = f'HSI: band - {q}', coloraxis_showscale=False)
        cls.update_xaxes(showticklabels=False)
        cls.update_yaxes(showticklabels=False)
        cls.show()
        plt.plot(img[:,:,q])
        plt.imshow(img[:,:,q], cmap='jet')
        plt.axis('off')
        plt.title(f'band - {q}')


    # Visualizing the Ground truth of the HSI

    cls = px.imshow(gt, color_continuous_scale='jet',)
                    
    cls.update_layout(title = 'Gound Trurh', coloraxis_showscale=False)
    cls.update_xaxes(showticklabels=False)
    cls.update_yaxes(showticklabels=False)
    cls.show()

    q = pd.concat([pd.DataFrame(data = dt), pd.DataFrame(data = gt.ravel())], axis = 1)
    q.columns = [f'PC-{i}' for i in range(1,nbands+1)]+['class']
    print(q.head())

    #eliminazione classe 0
    qq = q[q['class'] != 0]
    print(qq['class'].value_counts())

    #diamo un nome alle classi
    class_labels = {'1': 'Asphalt','2'  :'Meadows','3'  :'Gravel',
    '4' :'Trees','5':'Painted metal sheets','6' : 'Bare Soil',
    '7' :'Bitumen','8'  :'Self Blocking Bricks','9' :'Shadows'}

    print("Stampo dataset")
    qq['label'] = qq['class'].apply(lambda x : class_labels[str(x)])
    print(qq['label'].value_counts())
    print(qq.head())


    #visualizzare istogramma
    print("Stampo istogramma")
    count = qq['label'].value_counts()
    bar_fig = px.bar(x = count.index[1:], y = count[1:], labels= class_labels, color = count.index[1:] )
    bar_fig.update_layout(
        xaxis = dict(
            title='Class',
            tickmode = 'array',
            tickvals = count.index[1:].tolist(),
            tickangle = 45),
        yaxis = dict(
            title='count',),
        showlegend=False)

    bar_fig.show()

    print("Campionamento dataset")
    #sampling dataset
    sample_size = 200
    sample = qq.groupby('class').apply(lambda x: x.sample(sample_size))
    print(sample.head())

    #PAIR PLOT
    pair = px.scatter_matrix(sample, dimensions=["PC-1", "PC-2", "PC-3"], color="label")
    pair.show()
    #py.plot(pair, filename = 'pair_plot_pc', auto_open=True)

    #2D Scatter 
    fig = px.scatter(sample, x="PC-1", y="PC-2", size="class", color="label",
            hover_name="label", log_x=True, size_max=12)
    fig.show()

    print("3D Scatter Plot")
    scatter_3d = px.scatter_3d(sample, x="PC-1", y="PC-2", z="PC-3", color="label", size="class", hover_name="label",
                symbol="label")#, color_discrete_map = {"Joly": "blue", "Bergeron": "green", "Coderre":"red"})
    scatter_3d.show()
    #py.plot(scatter_3d, filename = 'scatter_3d', auto_open=True)


def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def applyPCA1(X,gt, numComponents, flag):
    print("Fase PCA...")
    newX = np.reshape(X, (-1, X.shape[2]))
    
    start = datetime.datetime.now()
    pca = PCA(n_components=numComponents, whiten=True)
    end = datetime.datetime.now()
    timePCA = end -start
    print("Tempo PCA (millisecondi): ")
    print(timePCA.microseconds/100)

    newX = pca.fit_transform(newX)
    
    if(flag==1):
        showPCA(X,newX,gt,numComponents)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    print("Fine PCA...")
    return newX, pca



dataset_names = [
    v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--dataset", type=str, default=None, choices=dataset_names, help="Dataset to use."
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Model to train. Available:\n"
    "SVM (linear), "
    "SVM_grid (grid search on linear, poly and RBF kernels), "
    "baseline (fully connected NN), "
    "hu (1D CNN), "
    "hamida (3D CNN + 1D classifier), "
    "lee (3D FCN), "
    "chen (3D CNN), "
    "li (3D CNN), "
    "he (3D CNN), "
    "luo (3D CNN), "
    "sharma (2D CNN), "
    "boulch (1D semi-supervised CNN), "
    "liu (3D semi-supervised CNN), "
    "mou (1D RNN)",
)
parser.add_argument(
    "--folder",
    type=str,
    help="Folder where to store the "
    "datasets (defaults to the current working directory).",
    default="./Datasets/",
)
parser.add_argument(
    "--cuda",
    type=int,
    default=-1,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument(
    "--restore",
    type=str,
    default=None,
    help="Weights to use for initialization, e.g. a checkpoint",
)

# Dataset options
group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument(
    "--training_sample",
    type=float,
    default=10,
    help="Percentage of samples to use for training (default: 10%%)",
)
group_dataset.add_argument(
    "--sampling_mode",
    type=str,
    help="Sampling mode" " (random sampling or disjoint, default: random)",
    default="random",
)
group_dataset.add_argument(
    "--train_set",
    type=str,
    default=None,
    help="Path to the train ground truth (optional, this "
    "supersedes the --sampling_mode option)",
)
group_dataset.add_argument(
    "--test_set",
    type=str,
    default=None,
    help="Path to the test set (optional, by default "
    "the test_set is the entire ground truth minus the training)",
)
# Training options
group_train = parser.add_argument_group("Training")
group_train.add_argument(
    "--epoch",
    type=int,
    help="Training epochs (optional, if" " absent will be set by the model)",
)
group_train.add_argument(
    "--patch_size",
    type=int,
    help="Size of the spatial neighbourhood (optional, if "
    "absent will be set by the model)",
)
group_train.add_argument(
    "--lr", type=float, help="Learning rate, set by the model if not specified."
)
group_train.add_argument(
    "--class_balancing",
    action="store_true",
    help="Inverse median frequency class balancing (default = False)",
)
group_train.add_argument(
    "--batch_size",
    type=int,
    help="Batch size (optional, if absent will be set by the model",
)
group_train.add_argument(
    "--test_stride",
    type=int,
    default=1,
    help="Sliding window step stride during inference (default = 1)",
)
# Data augmentation parameters
group_da = parser.add_argument_group("Data augmentation")
group_da.add_argument(
    "--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)"
)
group_da.add_argument(
    "--radiation_augmentation",
    action="store_true",
    help="Random radiation noise (illumination)",
)
group_da.add_argument(
    "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
)

parser.add_argument(
    "--with_exploration", action="store_true", help="See data exploration visualization"
)
parser.add_argument(
    "--download",
    type=str,
    default=None,
    nargs="+",
    choices=dataset_names,
    help="Download the specified datasets and quits.",
)
parser.add_argument(
        "--pca",
        type=int,
        default=-1,
        help="Specify PCA numeber component (defaults to -1, no PCA)",)

parser.add_argument(
        "--visualization",
        type=int,
        default=-1,
        help="Show image PCA(defaults to -1, no PCA IMAGE)",)


args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride
PCAnum = args.pca
flagVisual = args.visualization

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

viz = visdom.Visdom(env=DATASET + " " + MODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


hyperparams = vars(args)
# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)




if(PCAnum>=1):
    #K = 30 if dataset == 'IP' else 15
    img,pca = applyPCA1(img,gt,numComponents=PCAnum,flag = flagVisual)
    #X, y = createImageCubes(X, y, windowSize=windowSize)
    N_BANDS = PCAnum



#img = open_file("CublasPCA.mat")["X"]

#Normalization
img = np.asarray(img, dtype="float32")
#img = (img - np.min(img)) / (np.max(img) - np.min(img))



# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]
print(N_BANDS)

# Parameters for the SVM grid search
SVM_GRID_PARAMS = [
    {"kernel": ["rbf"], "gamma": [1e-1, 1e-2, 1e-3], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000]},
    {"kernel": ["poly"], "degree": [3], "gamma": [1e-1, 1e-2, 1e-3]},
]

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Instantiate the experiment based on predefined networks
hyperparams.update(
    {
        "n_classes": N_CLASSES,
        "n_bands": N_BANDS,
        "ignored_labels": IGNORED_LABELS,
        "device": CUDA_DEVICE,
        "pca": PCAnum,
    }
)
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

# # Show the image and the ground truth
# display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
# color_gt = convert_to_color(gt)





if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(
        img, gt, LABEL_VALUES, viz, ignored_labels=IGNORED_LABELS
    )
    plot_spectrums(mean_spectrums, viz, title="Mean spectrum/class")

results = []

# run the experiment several times
for run in range(N_RUNS):
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = open_file(TEST_GT)
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w, :h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
        # Sample random training spectra
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    print(
        "{} samples selected (over {})".format(
            np.count_nonzero(train_gt), np.count_nonzero(gt)
        )
    )
    print(
        "Running an experiment with the {} model".format(MODEL),
        "run {}/{}".format(run + 1, N_RUNS),
    )

    display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
    display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")
    
    if MODEL == "SVM_grid":
        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf = sklearn.model_selection.GridSearchCV(
            clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4
        )
        clf.fit(X_train, y_train)
        print("SVM best parameters : {}".format(clf.best_params_))
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        save_model(clf, MODEL, DATASET)
        startT = time.time()
        prediction = prediction.reshape(img.shape[:2])
        endT = time.time()
        print("Tempo Testing: ")
        timer(startT,endT)        
    elif MODEL == "SVM":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        startT = time.time()
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
        endT = time.time()
        print("Tempo Testing: ")
        timer(startT,endT) 
    elif MODEL == "SGD":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.linear_model.SGDClassifier(
            class_weight=class_weight, learning_rate="optimal", tol=1e-3, average=10
        )
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        startT = time.time()
        prediction = clf.predict(scaler.transform(img.reshape(-1, N_BANDS)))
        prediction = prediction.reshape(img.shape[:2])
        endT = time.time()
        print("Tempo Testing: ")
        timer(startT,endT) 
    elif MODEL == "nearest":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        class_weight = "balanced" if CLASS_BALANCING else None
        clf = sklearn.neighbors.KNeighborsClassifier(weights="distance")
        clf = sklearn.model_selection.GridSearchCV(
            clf, {"n_neighbors": [1, 3, 5, 10, 20]}, verbose=5, n_jobs=4
        )
        clf.fit(X_train, y_train)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        startT = time.time()
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
        endT = time.time()
        print("Tempo Testing: ")
        timer(startT,endT) 
    else:
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams["weights"] = torch.from_numpy(weights)
        # Neural network
        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
        # Split train set in train/val
        train_gt, val_gt = sample_gt(train_gt, 0.95, mode="random")
        # Generate the dataset
        train_dataset = HyperX(img, train_gt, **hyperparams)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=hyperparams["batch_size"],
            # pin_memory=hyperparams['device'],
            shuffle=True,
        )
        val_dataset = HyperX(img, val_gt, **hyperparams)
        val_loader = data.DataLoader(
            val_dataset,
            # pin_memory=hyperparams['device'],
            batch_size=hyperparams["batch_size"],
        )

        print(hyperparams)
        print("Network :")
        with torch.no_grad():
            for input, _ in train_loader:
                break
            summary(model.to(hyperparams["device"]), input.size()[1:])
            # We would like to use device=hyperparams['device'] altough we have
            # to wait for torchsummary to be fixed first.

        if CHECKPOINT is not None:
            model.load_state_dict(torch.load(CHECKPOINT))

        try:
            train(
                model,
                optimizer,
                loss,
                train_loader,
                hyperparams["epoch"],
                scheduler=hyperparams["scheduler"],
                device=hyperparams["device"],
                supervision=hyperparams["supervision"],
                val_loader=val_loader,
                display=viz,
            )
        except KeyboardInterrupt:
            # Allow the user to stop the training
            pass
        startT = time.time()
        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        endT = time.time()
        print("Tempo Testing: ")
        timer(startT,endT)

    run_results = metrics(
        prediction,
        test_gt,
        ignored_labels=hyperparams["ignored_labels"],
        n_classes=N_CLASSES,
    )

    mask = np.zeros(gt.shape, dtype="bool")
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    display_predictions(
        color_prediction,
        viz,
        gt=convert_to_color(test_gt),
        caption="Prediction vs. test ground truth",
    )
    
    

    results.append(run_results)
    show_results(run_results, viz, label_values=LABEL_VALUES)
    io.imsave("output.tif", color_prediction)

if N_RUNS > 1:
    show_results(results, viz, label_values=LABEL_VALUES, agregated=True)
    io.imsave("output.tif", color_prediction)
