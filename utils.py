import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from quicksave import QuickSaver

def seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def zero_grad(tensor):
    if tensor.grad is not None:
        tensor.grad.detach_()
        tensor.grad.zero_()

def save_results(qs, name, xs, ys):
    json = {n: (x, y) for n, (x, y) in enumerate(zip(xs, ys))}
    qs.save_json(json, name=name)

# def save_policies(name, p1s, p2s):
#     json = {
#         'p1 policy': p1s,
#         'p2 policy': p2s
#     }
#     QuickSaver().save_json(json, name=name)

def save_results_and_policies(qs, name, xs, ys, p1s, p2s):
    json = {n: (x, y, p1, p2) for n, (x, y, p1, p2) in enumerate(zip(xs, ys, p1s, p2s))}
    qs.save_json(json, name=name)

def load_results_policies(folder):
    dic = QuickSaver().load_json_path(os.path.join('quick_saves', folder, 'Pols_res_0.json'))
    r1s = [x[0] for k, x in dic.items()]
    r2s = [x[1] for k, x in dic.items()]
    p1s = [x[2] for k, x in dic.items()]
    p2s = [x[3] for k, x in dic.items()]
    return r1s, r2s, p1s, p2s

def load_cross_results(folder):
    dic = QuickSaver().load_json_path(os.path.join('quick_saves', folder, 'XPfs_0.json'))
    xr1s = [x[0] for k, x in dic.items()]
    xr2s = [x[1] for k, x in dic.items()]
    return xr1s, xr2s

def load_config(folder):
    dic = QuickSaver().load_json_path(os.path.join('quick_saves', folder, 'config_0.json'))
    return dic


# def plot_again(n, m, game):
#     xs, ys = load_results('Pfs_' + game + '_' + str(n) + '.json')
#     plot_results(xs, ys, color='orange')
#     xs, ys = load_results('XPfs_' + game + '_' + str(m) + '.json')
#     plot_results(xs, ys, game=game, save=game, color='blue')

def plot_results(env, xs, ys, color=None):
    nx = (0.5 - np.random.rand(len(xs))) * 0.2
    ny = (0.5 - np.random.rand(len(xs))) * 0.2
    xs = np.array(xs) + nx
    ys = np.array(ys) + ny
    if color is not None:
        sns.scatterplot(x=xs, y=ys, fc='none', ec=color, linewidth=1.3)
    else:
        sns.scatterplot(x=xs, y=ys)

    polygon = env.outcomes_polygon()
    plt.fill(polygon[0], polygon[1], alpha=0.2, color='purple')

def save_plot(qs, name):
    plt.savefig(os.path.join(qs.file_loc, name + '.png'))
    plt.clf()
