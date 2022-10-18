## Dependencies
```
Package                       Version
----------------------------- -------------------
click                         8.1.3
easydict                      1.10
matplotlib                    3.5.1
multiprocess                  0.70.12.2
numpy                         1.23.2
Pillow                        9.2.0
pip                           21.3.1
scikit-learn                  0.23.2
torch                         1.10.2+cu113
torchvision                   0.11.3+cu113
tqdm                          4.64.1
```
### Install Requirements
`pip install -r requirements.txt`


## Contents
```
code
│  classify.py : [Mission 3] : 일러스트 이미지 20종류 분류
│  cluster.py : [Mission 2] 실사와 일러스트 이미지 구분
│  dataset.py
│  loss.py
│  requirements.txt
│  scheduler.py
│  utils.py
│
├─models
│  │  AutoEncoder.py
│  │  BasicCNN.py
│  │
│  ├─BaseModel
│  │      model.py
│  │      __init__.py
│  │
│  └─EfficientNet
│          model.py
│          utils.py
│          __init__.py
│
└─output
        basemodel_0.5844.pth
        basemodel_0.6091.pth
```


## Code
```
python classify.py --outdir output --datadir ../data --model_name EfficientNet
```
### All arguments
```
@click.option('--outdir',       help='Where to save the results',           metavar='DIR',      type=str,           required=True)
@click.option('--datadir',      help='Data path',                           metavar='DIR',      type=str,           required=True)
@click.option('--model_name',   help='Model name to train',                 metavar='STR',      type=str,           required=True)

# Optional features.
@click.option('--resize',       help='How much to resize',                  metavar='INT',      type=click.IntRange(min=1),                 default=64)
@click.option('--batch_size',   help='Total batch size',                    metavar='INT',      type=click.IntRange(min=1),                 default=256)
@click.option('--epochs',       help='Epochs',                              metavar='INT',      type=click.IntRange(min=1),                 default=15)
@click.option('--fold',         help='Whether to apply cross fold',         metavar='BOOL',     type=bool,                                  default=False)
@click.option('--lr',           help='Learning rate',                       metavar='FLOAT',    type=click.FloatRange(min=0),               default=1e-2)
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
@click.option('--save_name',    help='Name of model when saved',            metavar='STR',      type=str,                                   default='experiment',    show_default=True)
@click.option('--save_limit',   help='# of saved models',                   metavar='INT',      type=click.IntRange(min=1),                 default=2,              show_default=True)
@click.option('--seed',         help='Random seed',                         metavar='INT',      type=click.IntRange(min=0),                 default=0,              show_default=True)
```
