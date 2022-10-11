
from models import *
from dataset import *
from utils import *

from easydict import EasyDict
from importlib import import_module

import click


@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results',       metavar='DIR',      type=str,           required=True)
@click.option('--data',         help='Data path',                       metavar='DIR',      type=str,           required=True)
@click.option('--record',       help='Tfrecord path',                   metavar='DIR',      type=str,           required=True)
@click.option('--model',        help='Model name to train',             metavar='STR',      type=str,           required=True)

# Optional features.
@click.option('--resize',       help='How much to resize',              metavar='INT',      type=click.IntRange(min=1),         default=64)
@click.option('--batch_size',   help='Total batch size',                metavar='INT',      type=click.IntRange(min=1),         default=256)
@click.option('--epochs',       help='Epochs',                          metavar='INT',      type=click.IntRange(min=1),         default=15)
@click.option('--fold',         help='Whether to apply cross fold',     metavar='BOOL',     type=bool,                          default=False)
@click.option('--cutmix',       help='Cutmix probability',              metavar='FLOAT',    type=click.FloatRange(min=0),       default=0)
@click.option('--val_ratio',    help='Proportion of valid dataset',     metavar='FLOAT',    type=click.FloatRange(min=0.1),     default=0.15)
@click.option('--test_ratio',   help='Proportion of test dataset',      metavar='FLOAT',    type=click.FloatRange(min=0.1),     default=0.15)
@click.option('--checkpoint',   help='checkpoint path',                 metavar='DIR',      type=str,                           default='')

# Misc settings.
@click.option('--save_name',    help='Name of saved model',             metavar='STR',      type=str,                           default='experiment',   show_default=True)
@click.option('--seed',         help='Random seed',                     metavar='INT',      type=click.IntRange(min=0),         default=0,              show_default=True)
@click.option('--workers',      help='DataLoader worker processes',     metavar='INT',      type=click.IntRange(min=1),         default=3,              show_default=True)

def main(**kwargs):
    ## 인자 입력받기 ##
    opts = EasyDict(kwargs)

    ## 난수 고정 ##
    set_seed(opts.seed)

    ## K-fold ##
    ''' utils.py에 추후 개발 필요'''

    ## 데이터 ##
    dataset = read_dataset(data_path=opts.data, record_path=opts.record, epochs=opts.epochs, batch_size=opts.batch_size, resize=opts.resize)
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset, opts.val_ratio, opts.test_ratio, shuffle=False)
    '''
    민영님!@
    기존에 tfrecord파일을 이미지 당 하나씩 개별적으로 저장하셨는데, 효율성을 좀 더 챙겨보고자
    하나의 tfrecord파일에 모든 데이터(image, label, name)를 저장해 두었습니다.
    다만, 걸리는 부분이 있는데, 인터넷에 나와있는 자료들은 dataset이
    (image, label) 형태로 되어있어서 바로 model.fit(dataset)이 가능한데,
    우리 dataset은 (image, label, name)으로 구성되어 있다보니까 이 부분이 좀 걸리네요.
    (image, name)으로 dataset 형태를 바꿔야 하는지,, 아니면 그냥 돌려도 되는지 잘 모르겠습니다! ㅜ-ㅜ
    그래서 이 부분을 해결해서 학습에 들어갈 수 있도록 해야할 것 같아요..(나중에 file name이랑도 연결되도록)
    
    마지막으로,
    시각화 부분은 간단히 구현해 놓긴했는데 아마 실행해보시면서 수정하셔야 할 것 같고,
    cutmix 부분은 dataset.py에 추가로 구현하셔서 적용하면 될 것 같아요!
    opts.cutmix 인자로 확률도 설정할 수 있도록 해놓았습니다.
    
    여기 classifiy.py에서는 가능하면 함수로만 실행될 수 있도록 구현했습니다!
    추가적으로 필요한 코드 있으면 자유롭게 구현하시면 될 것 같아요. 
    궁금하신 점 있으면 알려주세요!!
    '''

    ## 스케쥴러 ##
    ''' scheduler.py에 추후 개발 필요'''

    ## 모델 ##
    '''
    종원님!@
    지금 dataset을 완벽하게 세팅을 못해서 바로 학습하기는 어렵습니다..!
    
    모델은 종원님이 올려주신 모델 그대로 사용했고 models/BasicCNN.py에 구현해두었습니다.
    subclass 방식으로 모델을 구성해두었어요...! 지금껏 구현하신 방식이랑 좀 다르긴 할텐데 어렵진 않을거에요!
    
    앞으로 점점 생길 모델 클래스들은 한 파일에 담겨있으면 너무 지저분해질 것 같아서 models 폴더안에 여러 py 파일로 구성하도록 했어요
    현재는 실사 구분하는 encoder파일이랑 종원님 모델 구조 담긴 basiccnn파일로 있습니다.
    
    model = getattr(import_module(opts.model), "Model")(input_shape)
    아래쪽에 있는 이 코드는, 인자로 학습할 모델이 담겨있는 py파일 이름을 인자로 받고,
    그 후에 Model이라는 클래스를 가져온다는 뜻이니까.. 나중에 다른 모델 넣으실 때도
    클래스명은 Model로 하시기를 추천드립니다..!
    
    걸리는 부분은 model.build랑 summary를 잘 구현해 놓았는지 모르겠네요 ㅠㅠ
    그리고 utils.py에 있는 get_score 함수가지고 뒤쪽 성능만 얻으면 될 것 같아요!!
    
    여기 classifiy.py에서는 가능하면 함수로만 실행될 수 있도록 구현했습니다!
    추가적으로 필요한 코드 있으면 자유롭게 구현하시면 될 것 같아요. 
    궁금하신 점 있으면 알려주세요!!
    '''
    input_shape=(opts.resize, opts.resize, 3)
    model = getattr(import_module(opts.model), "Model")(input_shape)
    model.build(input_shape=input_shape)  # (1, feature)
    model.summary()

    ## 컴파일 및 학습 ##
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_dataset, epochs=20, verbose=1, validation_split=0.2, shuffle=True, batch_size=4)
    save_model(model, opts.outdir, opts.save_name)

    ## 성능 ##
    '''utils.py에 개발 필요'''
    get_score()

    ## 시각화
    ''' dataset.py의 show_batch 함수로 간단히 구현되어있음'''
    ''' 필요에 따라 추가 구현'''

    ## 인퍼런스
    '''추후 개발 필요'''


if __name__ == '__main__':
    main()

