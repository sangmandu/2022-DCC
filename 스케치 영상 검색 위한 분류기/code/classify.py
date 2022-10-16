
from dataset import *
from models import BasicCNN
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
@click.option('--epochs',       help='Epochs',                          metavar='INT',      type=click.IntRange(min=1),         default=30)
@click.option('--fold',         help='Whether to apply cross fold',     metavar='BOOL',     type=bool,                          default=False)
@click.option('--cutmix',       help='Cutmix probability',              metavar='FLOAT',    type=click.FloatRange(min=0),       default=0)
@click.option('--val_ratio',    help='Proportion of valid dataset',     metavar='FLOAT',    type=click.FloatRange(min=0.1),     default=0.15)
@click.option('--test_ratio',   help='Proportion of test dataset',      metavar='FLOAT',    type=click.FloatRange(min=0.1),     default=0.15)
@click.option('--checkpoint',   help='checkpoint path',                 metavar='DIR',      type=str,                           default='')
@click.option('--num_classes',  help='num_classes',                     metavar='INT',      type=click.IntRange(min=1),         default=20)

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
    # dataset = read_dataset(data_path=opts.data, record_path=opts.record, epochs=opts.epochs, batch_size=opts.batch_size, resize=opts.resize)
    # train_dataset, valid_dataset, test_dataset = split_dataset(dataset, opts.val_ratio, opts.test_ratio, shuffle=False)
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
##----------------------------------------------------------------------------
    ## 데이터 불러오기 ## 
    BasicCNN.make_npy(opts.resize, opts.resize, groups_folder_path = './data')## npy 파일 만드는 부분(1번만 해도 됨)
    x_train, x_test, y_train, y_test = np.load('./img_data.npy', allow_pickle=True)## npy 파일 읽는 부분
    ## 모델 ##
    model = BasicCNN.make_model(image_w = opts.resize, image_h = opts.resize)
    ## 컴파일 및 학습 ##
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    BasicCNN.fit_model(model,x_train, x_test, y_train, y_test,batch_size = opts.batch_size,epochs= opts.epochs,num_classes = opts.num_classes,model_name="BasicCNN")

    ## 성능 ##
    '''utils.py에 개발 필요'''
    model = BasicCNN.load_BasicCNN()
    BasicCNN.get_f1_score(model,x_test,y_test)
#------------------------------------------------------------------------------
    ## 시각화
    ''' dataset.py의 show_batch 함수로 간단히 구현되어있음'''
    ''' 필요에 따라 추가 구현'''

    ## 인퍼런스
    '''추후 개발 필요'''


if __name__ == '__main__':
    main()

