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

import multiprocessing
import click
import os
import wandb
from sklearn.metrics import classification_report
import warnings

from torch.utils.data import DataLoader, ConcatDataset

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

# Optional features.
@click.option('--resize',       help='How much to resize',                  metavar='INT',      type=click.IntRange(min=1),                 default=128)
@click.option('--batch_size',   help='Total batch size',                    metavar='INT',      type=click.IntRange(min=1),                 default=256)
@click.option('--epochs',       help='Epochs',                              metavar='INT',      type=click.IntRange(min=1),                 default=30)
@click.option('--fold',         help='Whether to apply cross fold',         metavar='BOOL',     is_flag=True)
@click.option('--lr',           help='Learning rate',                       metavar='FLOAT',    type=click.FloatRange(min=0),               default=5e-3)
@click.option('--optimizer',    help='Optimizer ',                          metavar='STR',      type=str,                                   default='AdamW')
@click.option('--scheduler',    help='Scheduler',                           metavar='STR',      type=str,                                   default='CyclicLR')
@click.option('--criterion',    help='Loss function',                       metavar='STR',      type=str,                                   default='cross_entropy')
@click.option('--val_ratio',    help='Proportion of valid dataset',         metavar='FLOAT',    type=click.FloatRange(),                    default=0.2)
@click.option('--checkpoint',   help='checkpoint path',                     metavar='DIR',      type=str,                                   default='')

# Image settings.
@click.option('--aug',          help='transform image or not',              metavar='FLOAT',    type=bool,                                  default=True)
@click.option('--cutmix',       help='Cutmix probability',                  metavar='FLOAT',    type=click.FloatRange(min=0, max=1),        default=0)
@click.option('--dup_sim',      help='threshold of duplicate similarity',   metavar='FLOAT',    type=click.FloatRange(min=0, max=1),        default=1)
@click.option('--sampling',     help='What sampling to apply',              metavar='STR',      type=str,                                   default='')
@click.option('--use_crop',     help='use crop image for enhancement',      metavar='BOOL',     is_flag=True)
@click.option('--check_stat',   help='computing image mean and std',        metavar='BOOL',     is_flag=True)
@click.option('--only_illust',  help='train only illustration image or not',metavar='BOOL',     type=bool,                                  default=True)
@click.option('--oversampling',     help='use oversampling for enhancement',      metavar='BOOL',     is_flag=True)
@click.option('--over_iter',  type=click.IntRange(min=0), multiple=True, default=[])
@click.option('--over_label', type=click.IntRange(min=0), multiple=True, default=[])



# Misc settings.
@click.option('--save_name',    help='Name of model when saved',            metavar='STR',      type=str,                                   default='experiment')
@click.option('--save_limit',   help='# of saved models',                   metavar='INT',      type=click.IntRange(min=1),                 default=2)
@click.option('--seed',         help='Random seed',                         metavar='INT',      type=click.IntRange(min=0),                 default=0)
@click.option('--use_wandb',    help='Wandb',                               metavar='BOOL',     is_flag=True)
@click.option('--f1_score_report',    help='f1_score_report',                               metavar='BOOL',     is_flag=True)



def main(**kwargs):
    ## Arguments
    opts = EasyDict(kwargs)
    print(opts)

    ## Random Seed
    set_seed(opts.seed)

    ## Wandb Settings
    if opts.use_wandb:
        wandb.init(
            project=opts.model_name,
            name=opts.save_name,
            config=opts,
            # reinit=True,
        )
    if not os.path.exists("./f1_score"):
        os.makedirs("./f1_score")
    ## Path Settings
    opts.save_name = check_paths(opts.outdir, opts.model_name, opts.save_name)

    ## Data
    df = load_data(opts.datadir, opts.dup_sim, opts.sampling, opts.use_crop, opts.only_illust)

    ## Loss Function
    criterion = create_criterion(opts.criterion)

    ## Model
    model = get_model(opts.model_name, opts.resize).to(DEVICE)

    ## Checkpoint
    if opts.checkpoint:
        print(f"checkpoint : {opts.checkpoint}")
        model.load_state_dict(torch.load(opts.checkpoint))

    ## Optimizer
    optimizer = get_optimizer(opts.optimizer, opts.lr, model) # default: Adam

    ## Scheduler
    base_step = int(len(df) * (1 - opts.val_ratio))
    # scheduler = get_scheduler(opts.scheduler, opts.lr, opts.batch_size, base_step, optimizer)
    scheduler = None

    ## Train
    print("Start training")
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
    train_df, eval_df = train_test_split(df, test_size=opts.val_ratio, stratify=df[['label']])

    ## Dataset
    i = 0
    if opts.oversampling:      # **oversampling 시킬건지에 대한 인자**

        train_df_over_need = train_df[train_df['label'].isin(opts.over_label)] # 라벨의 oversampling이 필요한 df    #**opts.arr는 oversampling이 필요한 라벨을 모아둔 list**
        train_df_over_not_need = train_df[~train_df['label'].isin(opts.over_label)] # 라벨 oversampling이 필요하지 않은 df

        train_dataset_over_need = Dataset(train_df_over_need, resize=opts.resize, aug=False)
        train_dataset_over_not_need = Dataset(train_df_over_not_need, resize=opts.resize, aug=True)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_over_need, train_dataset_over_not_need])

        for label in opts.over_label:
            tmp = train_df[train_df['label'] == label]

            for _ in range(opts.over_iter[i]):
                tmp_dataset = Dataset(tmp, resize=opts.resize, aug=True)
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, tmp_dataset])
            i += 1

    else:
        train_dataset = Dataset(train_df, resize=opts.resize, aug=opts.aug)

    valid_dataset = Dataset(eval_df, resize=opts.resize, aug=False)


    ## check image stats
    if opts.check_stat:
        mean, std = compute_mean_std(train_df, opts.resize)
        print(f"mean : {mean}")
        print(f"std : {std}")

        train_dataset.mean = valid_dataset.mean = mean
        train_dataset.std = valid_dataset.std = std

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


    ## train
    best_train_acc = best_valid_acc = 0
    best_train_loss = best_valid_loss = np.inf
    best_train_f1 = best_valid_f1 = 0

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
                f'Epoch #{epoch+1:2.0f} | '
                f'train | f1 : {train_batch_f1[-1]:.5f} | accuracy : {train_batch_accuracy[-1]:.5f} | '
                f'loss : {train_batch_loss[-1]:.5f} | lr : {get_lr(optimizer):.7f}'
            )

        # scheduler.step()

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
            if opts.f1_score_report:
                classification_report_y_pred = np.array([])
                classification_report_label = np.array([])

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

                if opts.f1_score_report:
                    classification_report_y_pred = np.append(classification_report_y_pred, preds.cpu().numpy().squeeze())  
                    classification_report_label = np.append(classification_report_label, labels.cpu().numpy().squeeze())

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
            with open(f"./f1_score/{opts.model_name}-f1 score_report-epoch{epoch}.txt", "w") as text_file:
               print(classification_report(classification_report_label, classification_report_y_pred), file=text_file)

            if cur_f1 >= best_valid_f1:
                if cur_f1 == best_valid_f1:
                    print(f"New best model for valid f1 : {cur_f1:.5%}! saving the best model..")
                    torch.save(model.state_dict(), f"{opts.save_name}_{cur_f1:.4f}.pth")
                    best_valid_f1 = cur_f1

                    if len(glob(f'{opts.save_name}_*.pth')) > opts.save_limit:
                        remove_item = sorted(glob(f'{opts.save_name}_*.pth'))[0]
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

            if opts.use_wandb:
                wandb.log({
                    "train_loss": train_item[0],
                    "train_acc": train_item[1],
                    "train_f1": train_item[2],
                    "best_train_f1": best_train_f1,

                    "valid_loss": valid_item[0],
                    "valid_acc": valid_item[1],
                    "valid_f1": valid_item[2],
                    "best_valid_f1": best_valid_f1,

                    "lr": get_lr(optimizer)
                })


if __name__ == '__main__':
    main()
