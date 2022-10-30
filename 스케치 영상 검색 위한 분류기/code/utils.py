from glob import glob
from tqdm import tqdm
from PIL import Image
from importlib import import_module

import subprocess as sp
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

def get_model(model_name, resize):
    if model_name == 'BaseModel':
        model_module = getattr(import_module('.'.join(['models', model_name, 'model'])), model_name)
        model = model_module(
            num_classes=20
        )
    elif model_name == 'EfficientNet':
        model_module = getattr(import_module('.'.join(['models', model_name, 'model'])), model_name)
        model = model_module.from_name(
            model_name='efficientnet-b0',
            image_size=resize,
            num_classes=20,
        )
    else:
        raise Exception("model name is incorrect")

    return model


def get_optimizer(optimizer_name, lr, model):
    opt_module = getattr(import_module("torch.optim"), optimizer_name)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=5e-4,
    )

    return optimizer


# https://sanghyu.tistory.com/113
def get_scheduler(scheduler_name, lr, batch_size, base_step, optimizer):
    sch_module = getattr(import_module("torch.optim.lr_scheduler"), scheduler_name)
    if scheduler_name == 'CyclicLR':
        scheduler = sch_module(
            optimizer,
            base_lr=5e-5,
            max_lr=lr,
            step_size_down=4,
            step_size_up=2,
            cycle_momentum=False,
            mode="triangular"
        )
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = sch_module(
            optimizer,
            T_max=100,
            eta_min=0
        )
    elif scheduler_name == 'St':
        scheduler = sch_module(
            optimizer,
            step_size=10,
            gamma=0.5
        )
    return scheduler


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def check_paths(outdir, model_name, save_name):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.exists(os.path.join(outdir, model_name)):
        os.makedirs(os.path.join(outdir, model_name))

    model_save_path = os.path.join(outdir, model_name, save_name)

    name_index = 1
    save_path_check = model_save_path
    while len(glob(save_path_check + '_*')) > 0:
        save_path_check = model_save_path
        save_path_check += str(name_index)
        name_index += 1
    model_save_path = save_path_check

    return model_save_path


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


# https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
def compute_mean_std(train_df, resize):
    mean = np.zeros(3)
    std = np.zeros(3)
    count = len(train_df)

    print("Computing both mean and std of images")
    for path in tqdm(train_df.path.values):
        image = np.array(Image.open(path).convert('RGB').resize((resize, resize)))
        mean += image.mean(axis=(0, 1))
        std += image.std(axis=(0, 1))

    return mean / count / 255, std / count / 255


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

