import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from quicksave import QuickSaver

###
# Maths-y stuff
###
def seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def zero_grad(tensor):
    if tensor.grad is not None:
        tensor.grad.detach_()
        tensor.grad.zero_()

###
# Saving and Loading
###
def save_results(qs, name, xs, ys):
    json = {n: (x, y) for n, (x, y) in enumerate(zip(xs, ys))}
    qs.save_json(json, name=name)

def load_results(folder, name):
    dic = QuickSaver().load_json_path(os.path.join('quick_saves', folder, name+'.json'))
    res1 = [x[0] for k, x in dic.items()]
    res2 = [x[1] for k, x in dic.items()]
    return res1, res2

def load_results_policies(folder):
    """
    Don't use anymore, left for backwards compatibility with some results
    """
    dic = QuickSaver().load_json_path(os.path.join('quick_saves', folder, 'Pols_res_0.json'))
    r1s = [x[0] for k, x in dic.items()]
    r2s = [x[1] for k, x in dic.items()]
    p1s = [x[2] for k, x in dic.items()]
    p2s = [x[3] for k, x in dic.items()]
    return r1s, r2s, p1s, p2s

def load_config(folder):
    dic = QuickSaver().load_json_path(os.path.join('quick_saves', folder, 'config_0.json'))
    return dic

def save_plot(qs, name):
    """
    Save the current plot and clear it.
    """
    plt.savefig(os.path.join(qs.file_loc, name + '.png'))
    plt.clf()
