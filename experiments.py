import random
import argparse

import torch
import numpy as np

from lola import lola_train
from utils import plot_results, save_results, load_results, plot_again, save_policies, seed
from test_utils import test, test_cross
from games import get_game

def lola_exp(game, gamma, lr, train_ep):
    """
    Given a game, initialise random agents, train with lola,
    and return average payoff for each step
    """
    env = get_game(game)()
    p1_a1 = torch.randn(5, requires_grad=True, dtype=torch.float64)
    p2_a1 = torch.randn(5, requires_grad=True, dtype=torch.float64)

    p1_a1, p2_a1 = lola_train(env, p1_a1, p2_a1, train_ep, gamma, lr)

    # Convert to probabilities
    p1_a1 = torch.sigmoid(p1_a1)
    p2_a1 = torch.sigmoid(p2_a1)

    r1, r2 = test(env, p1_a1, p2_a1)
    return p1_a1, p2_a1, r1, r2

def full_lola(iterations, gamma, lr, train_ep, sd):
    seed(sd)
    results = []
    policies = []
    for game in ['PD', 'BoS']:
        print("---- Starting {} ----".format(game))
        results.append([])
        policies.append([])
        for n in range(iterations):
            print("Iteration", n, end='\r')
            p1_a1, p2_a1, r1, r2 = lola_exp(game, gamma, lr, train_ep)
            results[-1].append((r1, r2))
            policies[-1].append((p1_a1, p2_a1))
        print()

        ### From standard play
        x = [r1 for r1, r2 in results[-1]]
        y = [r2 for r1, r2 in results[-1]]
        plot_results(x, y, color='orange')
        save_results('Pfs_' + game, x, y)

        ### Cross play
        p1s = [a.tolist() for a, b in policies[-1]]
        p2s = [b.tolist() for a, b in policies[-1]]
        save_policies('Pols_' + game, p1s, p2s)

        env = get_game(game)()
        r1_cross, r2_cross = test_cross(env, p1s, p2s)
        plot_results(r1_cross, r2_cross, game=game, save=game, color='blue')
        save_results('XPfs_' + game, r1_cross, r2_cross)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-e', default='lola', choices=['lola'])
    parser.add_argument('--iterations', '-i', default=40, type=int)
    parser.add_argument('--train_ep', '-te', default=100, type=int)
    parser.add_argument('--gamma', '-g', default=0.96, type=float)
    parser.add_argument('--learning_rate', '-lr', default=1, type=float)
    parser.add_argument('--seed', '-s', default=1234)
    args = parser.parse_args()

    if args.experiment == 'lola':
        full_lola(args.iterations, args.gamma, args.learning_rate, args.train_ep, args.seed)
