from datetime import datetime

from torchvision import transforms
from torchvision.transforms import *
from torch.utils.data import Dataset
import torch

from glob import glob
from tqdm import tqdm
from PIL import Image
from imagededup.methods import CNN

import cv2
import pandas as pd
import numpy as np
import os
import re


class Dataset(Dataset):
    def __init__(self, df, resize, mean=(0.8103, 0.7944, 0.7771), std=(0.2300, 0.2430, 0.2584), type='train'):
        self.df = df
        self.mean = mean
        self.std = std

        if type == 'train':
            self.transform = TrainAugmentation(resize, mean, std)
        else:
            self.transform = BaseAugmentation(resize, mean, std)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        name = self.df.name.iloc[index]

        path = self.df.path.iloc[index]
        image = Image.open(path).convert('RGB')
        image_transform = self.transform(image)

        label = self.df.label.iloc[index]

        return name, image_transform, label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class TrainAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize((resize, resize)),
            RandomChoice([ColorJitter(brightness=(0.2, 3)),
                          ColorJitter(contrast=(0.2, 3)),
                          ColorJitter(saturation=(0.2, 3)),
                          ColorJitter(hue=(-0.3, 0.3))]),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize((resize, resize)),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


def remove_duplicated_images(datadir, paths, dup_sim):
    cnn = CNN()
    directory = glob(os.path.join(datadir, '*'))
    dup_path = [] # 중복 이미지 path 리스트
    for i in tqdm(directory):
        encodings = cnn.encode_images(image_dir=i)
        duplicates = cnn.find_duplicates_to_remove(encoding_map = encodings, 
                                                   image_dir = i,
                                                   min_similarity_threshold = dup_sim)
        dup_tmp = list(map(lambda x: i + '/' + x, duplicates)) # 중복 이미지 전체 path
        dup_path.extend(dup_tmp)
    dup_path = list(map(lambda x: x.replace('\\', '/'), dup_path)) # 역슬래시 -> 슬래시
    paths = list(map(lambda x: x.replace('\\', '/'), paths)) # 역슬래시 -> 슬래시
    dup_remove_paths = [x for x in paths if x not in dup_path] # 최종 이미지 path 생성
    return dup_remove_paths


def sampling_images(paths):
    '''

    :param paths:
    :return:
    '''

    return paths


def suburb(path) :
    image = cv2.imread(path)
    image_gray = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    blur = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=0)
    ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    edged = cv2.Canny(blur, 10, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0   

    contours_xy = np.array(contours)

    x_min, x_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)

    y_min, y_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
            y_min = min(value)
            y_max = max(value)

    x = x_min
    y = y_min
    w = x_max-x_min
    h = y_max-y_min


    if (x==0 and w ==0) or (y==0 and h ==0):img_trim = image
    else: img_trim = image[y:y+h, x:x+w]

    cv2.imwrite(path, img_trim)



def load_data(datadir, dup_sim, sampling):
    '''
    데이터를 불러와 중복 이미지를 제거하고 샘플링을 적용한 뒤 dataframe으로 반환합니다.
    :param datadir:
    :param dup_sim:
    :param sampling:
    :return: df:
    '''

    labelobj = re.compile('L2_([0-9]+)')

    paths = [image_path for image_path in glob(os.path.join(datadir, '*', '*'))]

    only_illustration = True
    if only_illustration:
        paths = [path for path in paths if datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d') != '2022-09-28']

    label_to_num = {re.findall(labelobj, path)[0]: num for num, path in enumerate(glob(os.path.join(datadir, '*')))}

    if dup_sim < 1:
        paths = remove_duplicated_images(datadir, paths, dup_sim)

    if sampling is not None:
        paths = sampling_images(paths)

    names = [re.findall('[a-z]+[.][a-z]+', path)[0] for path in paths]
    labels = [label_to_num[re.findall(labelobj, path)[0]] for path in paths]

    df = pd.DataFrame(
        data=zip(names, paths, labels),
        columns=['name', 'path', 'label']
    )
    
    df['path'] = df['path'].map(lambda x: suburb(x))

    return df


# https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
def compute_mean_std():
    pass


def denormalize_image(inputs_np, mean, std):
    pass

