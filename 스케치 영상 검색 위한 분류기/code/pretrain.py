from tqdm import tqdm

from dataset import load_data, Dataset
from utils import get_model, set_seed

from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
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

class Dataset(Dataset):
    def __init__(self, df, resize):
        self.df = df

        self.transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.RandomResizedCrop(resize-30),
            transforms.RandomApply([
                transforms.ColorJitter(0.5, 0.5, 0.5)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # name = self.df.name.iloc[index]

        path = self.df.path.iloc[index]
        image = Image.open(path).convert('RGB')

        q = self.transform(image)
        k = self.transform(image)

        # label = self.df.label.iloc[index]

        return q, k


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings(action='ignore')

os.environ['WANDB_LOG_MODEL'] = 'true'
os.environ['WANDB_WATCH'] = 'all'
os.environ['WANDB_SILENT'] = "true"


@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results',           metavar='DIR',      type=str,           required=True)
@click.option('--datadir',      help='Data path',                           metavar='DIR',      type=str,           required=True)

# Optional features.
@click.option('--resize',       help='How much to resize',                  metavar='INT',      type=click.IntRange(min=1),                 default=128)
@click.option('--batch_size',   help='Total batch size',                    metavar='INT',      type=click.IntRange(min=1),                 default=50)
@click.option('--epochs',       help='Epochs',                              metavar='INT',      type=click.IntRange(min=1),                 default=100)
@click.option('--lr',           help='Learning rate',                       metavar='FLOAT',    type=click.FloatRange(min=0),               default=5e-4)

# Misc settings.
@click.option('--seed',         help='Random seed',                         metavar='INT',      type=click.IntRange(min=0),                 default=0           )
@click.option('--use_wandb',    help='Wandb',                               metavar='BOOL',     is_flag=True)

def main(**kwargs):
    ## Arguments
    opts = EasyDict(kwargs)
    print(opts)

    set_seed(opts.seed)

    df = load_data(datadir=opts.datadir, dup_sim=0.91, sampling='', crop=False, only_illust=True)
    dataset = Dataset(df, resize=opts.resize)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)

    q_encoder = get_model('EfficientNetV2', opts.resize)

    # define classifier for our task
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(q_encoder.output_channel, 100)),
        ('added_relu1', nn.ReLU()),
        ('fc2', nn.Linear(100, 50)),
        ('added_relu2', nn.ReLU()),
        ('fc3', nn.Linear(50, 25))
    ]))

    # replace classifier
    # and this classifier make representation have 25 dimention
    q_encoder.classifier = classifier

    # define encoder for key by coping q_encoder
    k_encoder = copy.deepcopy(q_encoder)

    # move encoders to device
    q_encoder = q_encoder.to(DEVICE)
    k_encoder = k_encoder.to(DEVICE)

    print(
        summary(q_encoder, (3, 224, 224), device=DEVICE)
    )

    return
    # define loss function
    def loss_func(q, k, queue, t=0.05):
        # t: temperature

        N = q.shape[0]  # batch_size
        C = q.shape[1]  # channel

        # bmm: batch matrix multiplication
        pos = torch.exp(torch.div(torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1), t))
        neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N, C), torch.t(queue)), t)), dim=1)

        # denominator is sum over pos and neg
        denominator = pos + neg

        return torch.mean(-torch.log(torch.div(pos, denominator)))

    # define optimizer
    opt = optim.AdamW(q_encoder.parameters(), lr=opts.lr, weight_decay=5e-4)

    # initialize the queue
    queue = None
    K = 8192  # K: number of negatives to store in queue

    # fill the queue with negative samples
    flag = 0
    if queue is None:
        while True:
            with torch.no_grad():
                print("making queue...")
                for img_q, img_k in dataloader:
                    # extract key samples
                    xk = img_k.to(DEVICE)
                    k = k_encoder(xk).detach()

                    if queue is None:
                        queue = k
                    else:
                        if queue.shape[0] < K:  # queue < 8192
                            queue = torch.cat((queue, k), 0)
                        else:
                            flag = 1  # stop filling the queue

                    if flag == 1:
                        break

            if flag == 1:
                break

    queue = queue[:K]
    # check queue
    print('number of negative samples in queue : ', len(queue))

    wandb.init(
        project='Pretraining',
        name='experiments',
        config=opts,
    )

    # training
    momentum = 0.999
    start_time = time.time()

    q_encoder.train()

    sanity_check = False
    for epoch in range(1, opts.epochs+1):
        print('Epoch {}/{}'.format(epoch, opts.epochs))

        q_encoder.train()
        running_loss = 0
        for img_q, img_k in tqdm(dataloader):
            # retrieve query and key
            xq = img_q.to(DEVICE)
            xk = img_k.to(DEVICE)

            # get model outputs
            q = q_encoder(xq)
            k = k_encoder(xk).detach()

            # normalize representations
            q = torch.div(q, torch.norm(q, dim=1).reshape(-1, 1))
            k = torch.div(k, torch.norm(k, dim=1).reshape(-1, 1))

            # get loss value
            loss = loss_func(q, k, queue)
            running_loss += loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            # update the queue
            queue = torch.cat((queue, k), 0)

            if queue.shape[0] > K:
                queue = queue[opts.batch_size:, :]

            # update k_encoder
            for q_params, k_params in zip(q_encoder.parameters(), k_encoder.parameters()):
                k_params.data.copy_(momentum * k_params + q_params * (1.0 - momentum))

        # store loss history
        epoch_loss = running_loss / len(dataloader)
        wandb.log({
            'loss': epoch_loss
        })

        print('train loss: %.6f, time: %.4f min' % (epoch_loss, (time.time() - start_time) / 60))


        # save weights
        if epoch % 2 == 0:
            torch.save(q_encoder.state_dict(), os.path.join('output', 'pretrain', f'ep{epoch}.pth'))

        if sanity_check:
            torch.save(q_encoder.state_dict(), os.path.join('output', 'pretrain', f'ep{epoch}.pth'))
            break

if __name__ == '__main__':
    main()