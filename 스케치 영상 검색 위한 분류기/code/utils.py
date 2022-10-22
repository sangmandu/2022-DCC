from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def grid_image(np_images, gts, preds, n=16, shuffle=False):
#     batch_size = np_images.shape[0]
#     assert n <= batch_size
#
#     choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
#     figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
#     plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
#     n_grid = np.ceil(n ** 0.5)
#     tasks = ["mask", "gender", "age"]
#     for idx, choice in enumerate(choices):
#         gt = gts[choice].item()
#         pred = preds[choice].item()
#         image = np_images[choice]
#         # title = f"gt: {gt}, pred: {pred}"
#         gt_decoded_labels = Dataset.decode_multi_class(gt)
#         pred_decoded_labels = Dataset.decode_multi_class(pred)
#         title = "\n".join([
#             f"{task} - gt: {gt_label}, pred: {pred_label}"
#             for gt_label, pred_label, task
#             in zip(gt_decoded_labels, pred_decoded_labels, tasks)
#         ])
#
#         plt.subplot(n_grid, n_grid, idx + 1, title=title)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(image, cmap=plt.cm.binary)

    # return figure


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def get_score():
    pass


def cutmix(inputs, labels):
    W = inputs.shape[2]
    H = inputs.shape[3]
    mix_ratio = np.random.beta(1, 1)
    cut_W = np.int(W * mix_ratio)
    cut_H = np.int(H * mix_ratio)
    bbx1 = np.random.randint(W - cut_W)
    bbx2 = bbx1 + cut_W
    bby1 = np.random.randint(H - cut_H)
    bby2 = bby1 + cut_H

    rand_index = torch.randperm(len(inputs))
    labels_a = labels # 원본 이미지 label
    labels_b = labels[rand_index] # 패치 이미지 label

    # inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[rand_index, :, bby1:bby2, bbx1:bbx2]
    inputs[:, :, bby1:bby2,bbx1:bbx2] = inputs[rand_index, :, bby1:bby2, bbx1:bbx2]

    return inputs, labels_a, labels_b, mix_ratio
