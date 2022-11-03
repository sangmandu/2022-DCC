from dataset import load_data, Dataset
from utils import *
from loss import create_criterion

from easydict import EasyDict

from glob import glob
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import DataLoader
import torch

import pandas as pd
import re
import multiprocessing
import click
import os
import warnings

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings(action='ignore')

os.environ['WANDB_LOG_MODEL'] = 'true'
os.environ['WANDB_WATCH'] = 'all'
os.environ['WANDB_SILENT'] = "true"


@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results',           metavar='DIR',      type=str,           required=True)
@click.option('--datadir',      help='Data path',                           metavar='DIR',      type=str,           required=True)
@click.option('--model_name',   help='Model name to train',                 metavar='STR',      type=str,           required=True)
@click.option('--checkpoint',   help='checkpoint name',                     metavar='DIR',      type=str,           required=True)

@click.option('--only_illust',  help='Include schetch data',                metavar='BOOL',     type=bool,          default=True)
@click.option('--resize',       help='How much to resize',                  metavar='INT',      type=click.IntRange(min=1),                 default=128)
@click.option('--batch_size',   help='Total batch size',                    metavar='INT',      type=click.IntRange(min=1),                 default=256)

def main(**kwargs):
    ## Arguments
    opts = EasyDict(kwargs)
    print(opts)

    labelobj = re.compile('L2_([0-9]+)')
    label_to_num = {re.findall(labelobj, path)[0]: num for num, path in enumerate(glob(os.path.join(opts.datadir, '*')))}
    print(label_to_num)

    paths = [image_path for image_path in glob(os.path.join(opts.datadir, '*', '*'))]
    if opts.only_illust:
        paths = [path for path in paths if 's_' not in path and 'p_' not in path]

    print(f"{len(paths)} data has been set")

    names = [re.findall('([i].+)[.]jpg|([i].+)[.]png', path)[0][0] for path in paths]
    labels = [label_to_num[re.findall(labelobj, path)[0]] for path in paths]

    df = pd.DataFrame(
        data=zip(names, paths, labels),
        columns=['name', 'path', 'label']
    )

    test_dataset = Dataset(df, resize=opts.resize, aug=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opts.batch_size,
        num_workers=multiprocessing.cpu_count() // 3,
        shuffle=False,
        pin_memory=True,
        # drop_last=True,
    )

    ## Model
    model = get_model(opts.model_name, opts.resize).to(DEVICE)
    checkpoint_path = os.path.join(opts.outdir, opts.model_name, opts.checkpoint)
    model.load_state_dict(torch.load(checkpoint_path))

    with torch.no_grad():
        model.eval()

        test_batch_accuracy = []
        test_batch_f1 = []

        # figure = None
        test_pbar = tqdm(test_loader, total=len(test_loader))
        for (names, inputs, labels) in test_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)

            test_batch_accuracy.append(
                (labels == preds).sum().item() / opts.batch_size
            )
            f1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='micro')
            test_batch_f1.append(
                f1
            )

            test_pbar.set_description(
                f'test | f1 : {test_batch_f1[-1]:.5f} | accuracy : {test_batch_accuracy[-1]:.5f} | '
            )

        print(f"{opts.model_name} | {opts.checkpoint} | f1 : {sum(test_batch_f1) / len(test_loader)}")


if __name__ == '__main__':
    main()