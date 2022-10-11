from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

import numpy as np
import os
import random
import tensorflow as tf


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(2)


def save_model(model, outdir, checkpoint):
    idx = 1
    save_name = os.path.join(outdir, checkpoint)
    while os.path.exists(save_name):
        save_name = os.path.join(outdir, checkpoint+str(idx))
        idx += 1
    model.save(save_name)


def get_score():
    pass
