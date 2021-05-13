import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from quicksave import QuickSaver
from  games import get_game

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
    xr1s = [x[0] for k, x in dic.items()]
    xr2s = [x[1] for k, x in dic.items()]
    return xr1s, xr2s

def load_config(folder):
    dic = QuickSaver().load_json_path(os.path.join('quick_saves', folder, 'config_0.json'))
    return dic

def save_plot(qs, name):
    """
    Save the current plot and clear it.
    """
    plt.savefig(os.path.join(qs.file_loc, name + '.png'))
    plt.clf()

###
# Plotting
###
def add_noise(xs, ys, mag=0.2):
    """
    Add a small amount of random noise to xs and ys.
    """
    nx = (0.5 - np.random.rand(len(xs))) * mag
    ny = (0.5 - np.random.rand(len(xs))) * mag
    xs = np.array(xs) + nx
    ys = np.array(ys) + ny
    return xs, ys

def plot_results(env, prior, xs, ys, color=None):
    """
    Given some xs and ys, plot them on a scatterplot with some noise.
    """
    xs, ys = add_noise(xs, ys)
    if color is not None:
        sns.scatterplot(x=xs, y=ys, fc='none', ec=color, linewidth=1.3)
    else:
        sns.scatterplot(x=xs, y=ys)

    polygon = env.outcomes_polygon(prior=prior)
    plt.fill(polygon[0], polygon[1], alpha=0.1, color='purple')

def plot_from_folder(folder, noise_mag=0.2):
    """
    Load results from folder and make scatterplot with 
    normal play and cross play.
    """
    r1s, r2s = load_results(folder, name='Pfs')
    xr1s, xr2s = load_results(folder, name='Xpfs')
    r1s, r2s = add_noise(r1s, r2s, mag=noise_mag)
    xr1s, xr2s = add_noise(xr1s, xr2s, mag=noise_mag)

    sns.scatterplot(x=r1s, y=r2s, fc='none', ec='orange', linewidth=1.3, label="Jointly Trained")
    sns.scatterplot(x=xr1s, y=xr2s, fc='none', ec='blue', linewidth=1.3, label="Cross Tests")

    config = load_config(folder)
    env = get_game(**config, max_steps=100)

    polygon = env.outcomes_polygon()
    plt.fill(polygon[0], polygon[1], alpha=0.1, color='purple')

### For plotting policies in one-shot games with two types
def oneshot_policy_coord(p):
    """
    AA: 0 AB: 1 BA: 2 BB: 3 Not obviously one of these: 4
    """
    coord = 0
    if p[0][4] < 0.05:
        coord += 2
    if p[1][4] < 0.05:
        coord += 1
    if (p[0][4] > 0.05 and p[0][4] < 0.95) or (p[1][4] > 0.05 and p[1][4] < 0.95):
        coord = 4
    return coord

def oneshot_policies_to_coords(ps):
    coords = [oneshot_policy_coord(p) for p in ps]
    return coords

def oneshot_coords_to_heatmap(c1s, c2s):
    hmap = [[0 for _ in range(5)] for _ in range(5)]
    for c1, c2 in zip(c1s, c2s):
        hmap[4-c1][c2] += 1
    return hmap

def plot_oneshot_policies(folder):
    """ 
    Makes a heatmap of policies. X-axis is all combinations of p1's policies for each type,
    Y-axis is p2
    """
    p1s, p2s = load_results(folder, name='Pols')
    c1s = oneshot_policies_to_coords(p1s)
    c2s = oneshot_policies_to_coords(p2s)
    hmap = oneshot_coords_to_heatmap(c1s, c2s)
    labels = ['AA', 'AB', 'BA', 'BB', 'None']
    cmap = sns.light_palette((260, 75, 60), input="husl", as_cmap=True)
    sns.heatmap(data=hmap, xticklabels=labels, yticklabels=list(reversed(labels)), cmap=cmap, linewidths=.5, annot=True)

### Plot non-crossplay results colour coded by p1's policy
def coord_to_strat(cs):
    labels = ['AA', 'AB', 'BA', 'BB', 'None']
    return [labels[c] for c in cs]

def plot_res_with_pol1(folder, noise_mag=0.2):
    """
    For plotting results colour-coded by the p1's policy
    This is for use with one-shot policies from mem1 games
    """
    r1s, r2s = load_results(folder, name='Pfs')
    p1s, _ = load_results(folder, name='Pols')
    c1s = oneshot_policies_to_coords(p1s)

    r1s, r2s = add_noise(r1s, r2s, mag=noise_mag)

    sns.scatterplot(x=r1s, y=r2s, hue=coord_to_strat(c1s), linewidth=1.3)

    config = load_config(folder)
    env = get_game(**config, max_steps=100)

    polygon = env.outcomes_polygon()
    plt.fill(polygon[0], polygon[1], alpha=0.1, color='purple')
