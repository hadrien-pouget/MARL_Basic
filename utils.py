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

def save_results(name, xs, ys):
    json = {
        'p1 payoff': xs,
        'p2 payoff': ys
    }
    QuickSaver().save_json(json, name=name)

def load_results(name):
    dic = QuickSaver().load_json(name)
    return dic['p1 payoff'], dic['p2 payoff']

def save_policies(name, p1s, p2s):
    json = {
        'p1 policy': p1s,
        'p2 policy': p2s
    }
    QuickSaver().save_json(json, name=name)

def load_policies(name):
    dic = QuickSaver().load_json(name)
    return dic['p1 policy'], dic['p2 policy']

def plot_again(n, m, game):
    xs, ys = load_results('Pfs_' + game + '_' + str(n) + '.json')
    plot_results(xs, ys, color='orange')
    xs, ys = load_results('XPfs_' + game + '_' + str(m) + '.json')
    plot_results(xs, ys, game=game, save=game, color='blue')

def plot_results(xs, ys, game=None, save=None, color=None):
    nx = (0.5 - np.random.rand(len(xs))) * 0.2
    ny = (0.5 - np.random.rand(len(xs))) * 0.2
    xs = np.array(xs) + nx
    ys = np.array(ys) + ny
    if color is not None:
        sns.scatterplot(x=xs, y=ys, fc='none', ec=color, linewidth=1.3)
    else:
        sns.scatterplot(x=xs, y=ys)

    polygons = {
        'PD': ([-3, -1, 0, -2], [0, -1, -3, -2]),
        'BoS': ([0, 2, 4], [0, 2, 1]),
    }
    if game in polygons:
        plt.fill(polygons[game][0], polygons[game][1], alpha=0.2, color='purple')

    if save is not None:
        plt.savefig(save + '.png')
        plt.clf()
