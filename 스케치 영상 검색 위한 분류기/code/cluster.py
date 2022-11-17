from datetime import datetime
from glob import glob

import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from dataset import load_data, Dataset
from utils import get_model, set_seed

from torchvision.transforms import *
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from easydict import EasyDict
from PIL import Image
from collections import OrderedDict

import time
import copy
import random
import numpy as np
import wandb
import warnings
import click
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings(action='ignore')

os.environ['WANDB_LOG_MODEL'] = 'true'
os.environ['WANDB_WATCH'] = 'all'
os.environ['WANDB_SILENT'] = "true"


@click.command()

# Required.
@click.option('--datadir',      help='Data path',                           metavar='DIR',      type=str,           required=True)

# Optional features.
@click.option('--resize',       help='How much to resize',                  metavar='INT',      type=click.IntRange(min=1),                 default=128)
@click.option('--batch_size',   help='Total batch size',                    metavar='INT',      type=click.IntRange(min=1),                 default=32)
@click.option('--epochs',       help='Epochs',                              metavar='INT',      type=click.IntRange(min=1),                 default=100)
@click.option('--lr',           help='Learning rate',                       metavar='FLOAT',    type=click.FloatRange(min=0),               default=5e-4)

# Misc settings.
@click.option('--seed',         help='Random seed',                         metavar='INT',      type=click.IntRange(min=0),                 default=0)

def main(**kwargs):
    ## Arguments
    opts = EasyDict(kwargs)
    print(opts)

    set_seed(opts.seed)

    df = load_data(datadir=opts.datadir, dup_sim=1, sampling='', crop=False, only_illust=False)

    dataset = Dataset(df=df, resize=opts.resize, aug=False)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False)

    model = get_model('EfficientNetV2', opts.resize).to(DEVICE)
    model.load_state_dict(torch.load('output/pretrain/ep100.pth'), strict=False)
    model.classifier = nn.Identity()

    paths = [image_path for image_path in glob('../data/*/*')]
    real_image_paths = [path for path in paths if datetime.fromtimestamp(os.path.getctime(path)).strftime('%Y-%m-%d') == '2022-09-28']

    encodings = []
    paths = []
    for step, (path, image, label) in tqdm(enumerate(dataloader, 1), total=len(dataloader)):
        image = image.to(DEVICE)
        encoding = model(image)
        encodings.extend([_encoding for _encoding in encoding.detach().cpu().numpy()])

        paths.extend([_path for _path in path])

    kmeans_cluster = KMeans(n_clusters=2)
    kmeans_cluster.fit(encodings)

    real_image_label = 0 if list(kmeans_cluster.labels_ == 1).count(True) > list(kmeans_cluster.labels_ == 0).count(True) else 1
    real_image_index = np.where(kmeans_cluster.labels_ == real_image_label)[0]

    precision = [paths[index] in real_image_paths for index in real_image_index].count(True) / len(real_image_index)
    recall = [paths[index] in real_image_paths for index in real_image_index].count(True) / len(real_image_paths)
    f1 = 2 * precision * recall / (precision + recall)

    print(f"found {len(real_image_index)} | precision {precision} | recall {recall} | f1 {f1}")



if __name__ == '__main__':
    main()








