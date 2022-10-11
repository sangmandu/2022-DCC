from tqdm import tqdm
from glob import glob

import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import os
import re


# 바이트 피처 함수
def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


# int 피처 함수
def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


# 이미지 파일 binary 형태 불러오는 함수
def read_imagebytes(data):
    file = open(data, 'rb')
    bytes = file.read()
    return bytes


def write_tfrecord(data_path, save_path):
    writer = tf.io.TFRecordWriter(save_path)

    for data in tqdm(glob(os.path.join(data_path, '*', '*'))):
        image_data = read_imagebytes(data)  # 이미지 파일 binary 형태 정보
        label = re.findall('[0-9]{2}', data)[0]  # 데이터 라벨 정보

        example = tf.train.Example(features=tf.train.Features(feature={
            # feature 정보 입력
            'image': bytes_feature(image_data),
            'label': int64_feature([int(label)]),
            'name': bytes_feature(re.findall('[a-z]*[.][a-z]*', data)[0]),
        }))

        writer.write(example.SerializeToString())


def read_dataset(data_path, record_path, epochs, batch_size, resize):
    AUTO = tf.data.experimental.AUTOTUNE

    if not os.path.exists(record_path):
        write_tfrecord(data_path, record_path)

    dataset = tf.data.TFRecordDataset(record_path)
    dataset = dataset.map(lambda x: _parse_image_function(x, resize), num_parallel_calls=AUTO)
    dataset = dataset.prefetch(10)
    dataset = dataset.repeat(epochs)
    dataset = dataset.shuffle(buffer_size=10 * batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def _parse_image_function(example, resize):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'name': tf.io.FixedLenFeature([], tf.string),
    }

    features = tf.io.parse_single_example(example, image_feature_description)
    image = tf.io.decode_raw(features['image'], tf.uint8)
    image = transform(image)
    image = tf.reshape(image, [resize, resize, 3])

    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, 20)

    name = tf.cast(features['name'], tf.string)

    return image, label, name


def split_dataset(dataset, val_ratio=0.15, test_ratio=0.15, shuffle=True):
    if shuffle:
        dataset = dataset.shuffle()

    dataset_size = sum(1 for _ in dataset)
    val_size = int(dataset_size * val_ratio)
    test_size = int(dataset_size * test_ratio)

    train_dataset, valid_dataset = dataset.skip(val_size), dataset.take(val_size)
    train_dataset, test_dataset = train_dataset.skip(test_size), train_dataset.take(test_size)

    return train_dataset, valid_dataset, test_dataset


def transform(image, resize):
    image = tf.image.resize(image, (resize, resize))
    image = tf.image.random_crop(image, (112, 112, 3))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.6, 1.4)
    image = tf.image.random_brightness(image, 0.4)
    image = image / 255

    return image


def show_batch(dataset):
    image_batch, _, _ = next(iter(dataset))
    image_batch = image_batch[:25]

    plt.figure(figsize=(15, 15))
    for n in range(25):
        plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        plt.axis("off")
    plt.show()

