import os
import time
from pathlib import Path

import wandb

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from sklearn import metrics

import cv2

import torch
from torch import load
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from model import Net
from data_loader import SurgVisDataset

def evaluate(testloader, model, dev):
    
    ground_truth = []
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)

            # Get outputs
            outputs = model(inputs)
            # Get predictions
            pred = outputs.max(1, keepdim=True)[1]
            
            ground_truth += labels.tolist()
            predictions += pred.t().tolist()[0]

    return ground_truth, predictions

if __name__ == "__main__":
    #wandb.init(project="surgvisdom")

    PATH = Path('train_1')
    PATH_PORCINE_1 = PATH.joinpath('Porcine')

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config          # Initialize config
    config.batch_size = 128        # input batch size for training (default: 64)
    # config.test_batch_size = 10    # input batch size for testing (default: 1000)
    config.epochs = 50             # number of epochs to train (default: 10)
    config.lr = 0.1               # learning rate (default: 0.01)
    config.momentum = 0.1          # SGD momentum (default: 0.5) 
    config.no_cuda = False         # disables CUDA training
    config.seed = 42               # random seed (default: 42)
    # config.log_interval = 10     # how many batches to wait before logging training status
    config.use_classes_weights = True
    config.crop_size = (420, 630)
    config.resize_shape = (256, 256)
    config.dataset_path = PATH_PORCINE_1

    train_transform = transforms.Compose([transforms.CenterCrop(config.crop_size),
                                            transforms.Resize(config.resize_shape),
                                            transforms.ToTensor()])

    dataset = SurgVisDataset(config.dataset_path, classes={'Knot_Tying':0, 'Needle_Driving':1, 'Dissection': 2}, transform=train_transform, verbose=False)

    n = len(dataset)
    n_train = int(n * 0.80)
    n_val = int(n * 0.1)
    n_dev_test = int(n * 0.1)
    print("dataset len:", n)

    # Always split using the same seed
    torch.manual_seed(config.seed)
    train_set, val_set, dev_test_set = random_split(dataset, (n_train, n_val, n_dev_test))
    torch.manual_seed(torch.initial_seed())

    print("Train set:", len(train_set))
    print("Val set:", len(val_set))
    print("Test set", len(dev_test_set))

    model = Net()
    weight_path = Path("Models/balmy-lion-10/model_epoch_13_loss_0.0481.h5")
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('dev:', dev)
    model.to(dev)

    # Log metrics with wandb
    #wandb.watch(model)

    #dataloader_train = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=8)
    dataloader_test = DataLoader(dev_test_set, batch_size=config.batch_size, shuffle=False, num_workers=2)

    ground_truth, predictions = evaluate(dataloader_test, model, dev)
    print(len(ground_truth), len(predictions))
    #import pdb; pdb.set_trace()

    # get list of classes
    inv_dict = {v:k for k, v in dataloader_test.dataset.dataset.classes.items()}
    labels = [inv_dict[i] for i in range(len(inv_dict.keys()))]

    print(metrics.classification_report(ground_truth, predictions, target_names=labels))

    ax= plt.subplot()
    sns.heatmap(metrics.confusion_matrix(ground_truth, predictions), annot=True, ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.show()


    #confusion_matrix_graph = confusion_matrix(y_test, predictions)
    #wandb.log({"pr_curve" : wandb.plot.pr_curve(ground_truth, predictions, labels=labels),
    #           "roc_curve": wandb.plot.roc_curve(ground_truth, predictions, labels=labels)})