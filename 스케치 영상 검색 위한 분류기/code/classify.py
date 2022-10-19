from dataset import load_data, Dataset
from utils import *
from loss import create_criterion

from easydict import EasyDict
from importlib import import_module
from glob import glob
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold

import multiprocessing
import click
import os

from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import DataLoader
import torch
import warnings

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings(action='ignore')


@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results',           metavar='DIR',      type=str,           required=True)
@click.option('--datadir',      help='Data path',                           metavar='DIR',      type=str,           required=True)
@click.option('--model_name',   help='Model name to train',                 metavar='STR',      type=str,           required=True)

# Optional features.
@click.option('--resize',       help='How much to resize',                  metavar='INT',      type=click.IntRange(min=1),                 default=64)
@click.option('--batch_size',   help='Total batch size',                    metavar='INT',      type=click.IntRange(min=1),                 default=256)
@click.option('--epochs',       help='Epochs',                              metavar='INT',      type=click.IntRange(min=1),                 default=15)
@click.option('--fold',         help='Whether to apply cross fold',         metavar='BOOL',     type=bool,                                  default=False)
@click.option('--lr',           help='Learning rate',                       metavar='FLOAT',    type=click.FloatRange(min=0),               default=1e-1)
@click.option('--optimizer',    help='Optimizer ',                          metavar='STR',      type=str,                                   default='Adam')
@click.option('--scheduler',    help='Scheduler',                           metavar='STR',      type=str,                                   default='CyclicLR')
@click.option('--criterion',    help='Loss function',                       metavar='STR',      type=str,                                   default='cross_entropy')
@click.option('--val_ratio',    help='Proportion of valid dataset',         metavar='FLOAT',    type=click.FloatRange(min=0.1),             default=0.15)
@click.option('--test_ratio',   help='Proportion of test dataset',          metavar='FLOAT',    type=click.FloatRange(min=0.1),             default=0.15)
@click.option('--checkpoint',   help='checkpoint path',                     metavar='DIR',      type=str,                                   default='')

# Image settings.
@click.option('--aug',          help='transform image or not',              metavar='FLOAT',    type=bool,                                  default=False)
@click.option('--cutmix',       help='Cutmix probability',                  metavar='FLOAT',    type=click.FloatRange(min=0, max=1),        default=0)
@click.option('--dup_sim',      help='threshold of duplicate similarity',   metavar='FLOAT',    type=click.FloatRange(min=0, max=1),        default=1)
@click.option('--sampling',     help='What sampling to apply',              metavar='STR',      type=str,                                   default='')

# Misc settings.
@click.option('--save_name',    help='Name of model when saved',            metavar='STR',      type=str,                                   default='experiment',   show_default=True)
@click.option('--save_limit',   help='# of saved models',                   metavar='INT',      type=click.IntRange(min=1),                 default=2,              show_default=True)
@click.option('--seed',         help='Random seed',                         metavar='INT',      type=click.IntRange(min=0),                 default=0,              show_default=True)
# @click.option('--workers',      help='DataLoader worker processes',         metavar='INT',      type=click.IntRange(min=1),                 default=3,              show_default=True)


def main(**kwargs):
    ## Arguments
    opts = EasyDict(kwargs)
    print(opts)

    ## Random Seed
    set_seed(opts.seed)

    ## Data
    df = load_data(opts.datadir, opts.dup_sim, opts.sampling)

    ## Loss Function
    if opts.criterion == 'weight_cross_entropy':
        criterion = create_criterion(
            criterion_name=opts.criterion,
            weight=torch.FloatTensor([]).to(DEVICE),
            reduction='mean',
        )
    else:
        criterion = create_criterion(criterion_name=opts.criterion)

    ## Model
    if opts.model_name == 'BaseModel':
        model_module = getattr(import_module('.'.join(['models', opts.model_name, 'model'])), opts.model_name)
        model = model_module(
            num_classes=20
        ).to(DEVICE)
    elif opts.model_name == 'EfficientNet':
        model_module = getattr(import_module('.'.join(['models', opts.model_name, 'model'])), opts.model_name)
        model = model_module.from_name(
            model_name='efficientnet-b0',
            image_size=opts.resize,
            num_classes=20,
        ).to(DEVICE)
    else:
        raise Exception("model name is incorrect")

    if opts.checkpoint:
        model.load_state_dict(torch.load(opts.checkpoint))


    ## Optimizer
    opt_module = getattr(import_module("torch.optim"), opts.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opts.lr,
        # weight_decay=5e-4,
    )


    ## Scheduler
    ''' scheduler.py에 추후 개발 필요'''
    base_step = int(len(df) * (1-opts.val_ratio-opts.test_ratio))
    scheduler = CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=opts.lr,
        step_size_down=base_step * 2 // opts.batch_size,
        step_size_up=base_step // opts.batch_size,
        cycle_momentum=False,
        mode="triangular")

    ## Train
    train(df, model, criterion, optimizer, scheduler, opts)


def train(df, model, criterion, optimizer, scheduler, opts):
    '''

    :param df:
    :param model:
    :param criterion:
    :param optimizer:
    :param opts:
    :param paths:
    :return:
    '''

    ## Wandb
    '''추후 구현'''

    ## Stratified K-Fold
    ''' 추후 구현'''
    train_df, eval_df = train_test_split(df, test_size=opts.val_ratio + opts.test_ratio, stratify=df[['label']])
    valid_df, test_df = train_test_split(eval_df, test_size=opts.val_ratio, stratify=eval_df[['label']])

    ## Dataset
    train_dataset = Dataset(train_df, resize=opts.resize, type='train')
    valid_dataset = Dataset(valid_df, resize=opts.resize, type='eval')
    test_dataset = Dataset(test_df, resize=opts.resize, type='eval')

    ## DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size,
        num_workers=multiprocessing.cpu_count() // 3,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=opts.batch_size,
        num_workers=multiprocessing.cpu_count() // 3,
        shuffle=False,
        pin_memory=True,
        # drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opts.batch_size,
        num_workers=multiprocessing.cpu_count() // 3,
        shuffle=False,
        pin_memory=True,
        # drop_last=True,
    )


    best_train_acc = best_valid_acc = 0
    best_train_loss = best_valid_loss = np.inf
    best_train_f1 = best_valid_f1 = 0

    if not os.path.exists(opts.outdir):
        os.makedirs(opts.outdir)

    if not os.path.exists(os.path.join(opts.outdir, opts.model_name)):
        os.makedirs(os.path.join(opts.outdir, opts.model_name))

    model_save_path = os.path.join(opts.outdir, opts.model_name, opts.save_name)

    name_index = 1
    save_path_check = model_save_path
    while len(glob(save_path_check + '_*')) > 0:
        save_path_check = model_save_path
        save_path_check += str(name_index)
        name_index += 1

    model_save_path = save_path_check

    for epoch in range(opts.epochs):
        model.train()

        train_batch_loss = []
        train_batch_accuracy = []
        train_batch_f1 = []

        train_pbar = tqdm(train_loader, total=len(train_loader))
        for idx, (names, inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if np.random.random() < opts.cutmix:
                inputs, labels_a, labels_b, mix_ratio = cutmix(inputs, labels)
                outs = model(inputs)
                loss = criterion(outs, labels_a) * mix_ratio + criterion(outs, labels_b) * (1. - mix_ratio)
            else:
                outs = model(inputs)
                loss = criterion(outs, labels)

            preds = torch.argmax(outs, dim=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_batch_loss.append(
                loss.item()
            )
            train_batch_accuracy.append(
                (preds == labels).sum().item() / opts.batch_size
            )
            f1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='micro')
            train_batch_f1.append(
                f1
            )

            train_pbar.set_description(
                f'Epoch #{epoch:2.0f} | '
                f'train | f1 : {train_batch_f1[-1]:.5f} | accuracy : {train_batch_accuracy[-1]:.5f} | '
                f'loss : {train_batch_loss[-1]:.5f} | lr : {get_lr(optimizer):.7f}'
            )

        train_item = (sum(train_batch_loss) / len(train_loader),
                      sum(train_batch_accuracy) / len(train_loader),
                      sum(train_batch_f1) / len(train_loader))
        best_train_loss = min(best_train_loss, train_item[0])
        best_train_acc = max(best_train_acc, train_item[1])
        best_train_f1 = max(best_train_f1, train_item[2])

        with torch.no_grad():
            model.eval()

            valid_batch_loss = []
            valid_batch_accuracy = []
            valid_batch_f1 = []

            # figure = None
            valid_pbar = tqdm(valid_loader, total=len(valid_loader))
            for (names, inputs, labels) in valid_pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                valid_batch_loss.append(
                    criterion(outs, labels).item()
                )
                valid_batch_accuracy.append(
                    (labels == preds).sum().item() / opts.batch_size
                )
                f1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='micro')
                valid_batch_f1.append(
                    f1
                )

                valid_pbar.set_description(
                    f'valid | f1 : {valid_batch_f1[-1]:.5f} | accuracy : {valid_batch_accuracy[-1]:.5f} | '
                    f'loss : {valid_batch_loss[-1]:.5f} | lr : {get_lr(optimizer):.7f}'
                )

                # 시각화를 위해 나중에 개발
                # if figure is None:
                #     inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                #     inputs_np = valid_dataset.denormalize_image(inputs_np, valid_dataset.mean, valid_dataset.std)
                #     figure = grid_image(
                #         inputs_np, labels, preds, n=16, shuffle=True
                #     )

            valid_item = (sum(valid_batch_loss) / len(valid_loader),
                          sum(valid_batch_accuracy) / len(valid_loader),
                          sum(valid_batch_f1) / len(valid_loader))
            best_valid_loss = min(best_valid_loss, valid_item[0])
            best_valid_acc = max(best_valid_acc, valid_item[1])
            best_valid_f1 = max(best_valid_f1, valid_item[2])
            cur_f1 = valid_item[2]

            if cur_f1 >= best_valid_f1:
                if cur_f1 == best_valid_f1:
                    print(f"New best model for valid f1 : {cur_f1:.5%}! saving the best model..")
                    torch.save(model.state_dict(), f"{model_save_path}_{cur_f1:.4f}.pth")
                    best_valid_f1 = cur_f1

                    if len(glob(f'{model_save_path}_*.pth')) > opts.save_limit:
                        remove_item = sorted(glob(f'{model_save_path}_*.pth'))[0]
                        os.remove(remove_item)

            print(
                f"[Train] f1 : {train_item[2]:.5}, best f1 : {best_train_f1:.5} || " 
                f"acc : {train_item[1]:.5%}, best acc: {best_train_acc:.5%} || "
                f"loss : {train_item[0]:.5}, best loss: {best_train_loss:.5} || "
            )
            print(
                f"[Valid] f1 : {valid_item[2]:.5}, best f1 : {best_valid_f1:.5} || "
                f"acc : {valid_item[1]:.5%}, best acc: {best_valid_acc:.5%} || "
                f"loss : {valid_item[0]:.5}, best loss: {best_valid_loss:.5} || "
            )
            print()


if __name__ == '__main__':
    main()


