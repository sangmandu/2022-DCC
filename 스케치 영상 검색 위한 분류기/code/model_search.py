import pickle
import time
import warnings

import wandb

from models.EfficientNetV2.model import EfficientNetV2
from utils import *
from dataset import load_data, Dataset

from numba import cuda
from tqdm import tqdm
from easydict import EasyDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim.lr_scheduler

import multiprocessing
import optuna

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
warnings.filterwarnings(action='ignore')

class MyTrial:
    i = 1


def train(model, train_loader, valid_loader, loss_fn, optimizer, epochs, cfgs, study_name):
    cfgs_log = {k+str(i): v for i, cfg in enumerate(cfgs, 1) for k, v in zip(['ex', 'c', 'r', 's', 'se'], cfg)}
    wandb.init(
        project=study_name,
        name=str(MyTrial.i),
        config=cfgs_log,
        reinit=True,
    )
    print(f"#{MyTrial.i}")
    MyTrial.i += 1

    best_f1 = 0
    for epoch in tqdm(range(epochs)):
        model.train()
        # train_pbar = tqdm(train_loader, total=len(train_loader))
        for names, inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outs = model(inputs)
            loss = loss_fn(outs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_f1 = []
        batch_loss = []
        with torch.no_grad():
            model.eval()
            # valid_pbar = tqdm(valid_loader, total=len(valid_loader))
            for names, inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                f1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='micro')
                batch_f1.append(f1)
                batch_loss.append(loss_fn(outs, labels).item())

            valid_loss = sum(batch_loss) / len(valid_loader)
            valid_f1 = sum(batch_f1) / len(valid_loader)
            best_f1 = max(best_f1, valid_f1)

            wandb.log({
                "valid_loss": valid_loss,
                "valid_f1": valid_f1,
                "best_f1": best_f1
            })

    return best_f1


def search_model(trial):
    num_stages = trial.suggest_int("num_stages", low=2, high=7, step=1)
    cfgs = []
    ex_repeat = ex_channel = 0
    for depth in range(2, num_stages+2):
        expand_ratio = 2 if depth == 2 else 4 if (num_stages + 2) // 2 + 1 >= depth else 6
        channel = trial.suggest_int(f'stage{depth}_channel', low=max(ex_channel, 12*depth), high=18*depth, step=6)
        ex_channel = channel

        repeat = trial.suggest_int(f'stage{depth}_repeat', low=max(1, ex_repeat), high=depth, step=1)
        ex_repeat = repeat

        stride = trial.suggest_int(f'stage{depth}_stride', low=1, high=2, step=1)
        se = 0 if (num_stages + 2) // 2 + 1 > depth else 1

        cfgs.append([expand_ratio, channel, repeat, stride, se])

    print(cfgs)
    model = EfficientNetV2(cfgs).to(device)
    return cfgs, model


def check_model():
    num_stages = 7
    cfgs = []
    for depth in range(2, num_stages+2):
        expand_ratio = 2 if depth == 2 else 4 if (num_stages + 2) // 2 + 1 >= depth else 6
        channel = 18*depth
        repeat = depth
        stride = 2 if depth % 2 else 1
        se = 0 if (num_stages + 2) // 2 + 1 > depth else 1
        cfgs.append([expand_ratio, channel, repeat, stride, se])

    try:
        model = EfficientNetV2(cfgs).to(device)
        return True
    except:
        return False


def objective(trial, opts, train_loader, valid_loader):
    cfgs, model = search_model(trial)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-4)

    f1 = train(model, train_loader, valid_loader, loss_fn, optimizer, opts.epochs, cfgs, opts.study_name)
    return f1


def main():
    opts = EasyDict(
        image_size=32,
        batch_size=64,
        classes=20,
        max_depth=10,
        epochs=20,
        seed=0,
        trial=100,
        aug=True,
        study_name='automl_search',
        load_study='',
    )

    set_seed(opts.seed)
    df = load_data('../data', 0.91, '', False, True)
    df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(200, len(x))))

    train_df, eval_df = train_test_split(df, test_size=0.1, stratify=df[['label']])
    print(f"df {len(df)} | train_df {len(train_df)} | eval_df {len(eval_df)}")
    print(df.groupby('label').apply(lambda x: len(x)))

    train_dataset = Dataset(train_df, resize=opts.image_size, aug=opts.aug)
    valid_dataset = Dataset(eval_df, resize=opts.image_size, aug=False)
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
    )

    if opts.load_study:
        print("study continue")
        with open('optuna/'+opts.load_study, 'rb') as file:
            study = pickle.load(file)
        MyTrial.i = int(opts.load_study[0]) + 1
    else:
        study = optuna.create_study(
            direction='maximize',
            study_name=opts.study_name
        )

    print(f"start at {MyTrial.i} with {opts.load_study}")
    save_path = 'optuna_search_model.pickle'

    model_available = check_model()
    print(model_available)
    assert model_available, "CUDA Out of Memory "

    for n in range(MyTrial.i, opts.trial):
        study.optimize(lambda trial: objective(trial, opts, train_loader, valid_loader), n_trials=1)

        with open('optuna/'+str(n)+save_path, 'wb') as file:
            pickle.dump(study, file)

    try:
        pruned_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        complete_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
    except:
        pass

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"{key}:{value}")


if __name__ == '__main__':
    main()
