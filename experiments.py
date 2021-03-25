import random
import argparse

import torch
import numpy as np

from utils import plot_results, save_results, load_results, plot_again, save_policies, seed, save_results_and_policies
from test_utils import test_iterated, test_cross_iterated, incomplete_oneshot_test
from games import get_game
from train_loop import train
from train_steps import naive_step, lola_step
from value_functions import get_value_incomplete_oneshot, get_value_iterated

def train_test_incomplete_oneshot(env, step_func, gamma, lr, train_ep):
    """
    Given a joint dist over the games, initialise random agents, train,
    and return average payoff for each step
    """
    # env = get_game(game)(dist_n)

    p1 = torch.randn(2, requires_grad=True)
    p2 = torch.randn(2, requires_grad=True)

    p1, p2 = train(env, p1, p2, get_value_incomplete_oneshot, step_func, train_ep, lr)

    # Convert to probabilities
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)

    r1, r2 = incomplete_oneshot_test(env, p1, p2, test_e=100)
    return p1, p2, r1, r2

def train_test_iterated(env, step_func, gamma, lr, train_ep)
    """
    Given a game, initialise random agents, train with lola,
    and return average payoff for each step
    """
    # env = get_game(game)()
    p1 = torch.randn(5, requires_grad=True, dtype=torch.float64)
    p2 = torch.randn(5, requires_grad=True, dtype=torch.float64)

    p1, p2 = train(env, p1, p2, get_value_iterated, step_func, train_ep, gamma, lr)

    # Convert to probabilities
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)

    r1, r2 = test(env, p1, p2)
    return p1, p2, r1, r2

def experiment(game, iterations, gamma, lr, train_ep):
    results = []
    policies = []

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


def full_incomplete_simple_lola(dist_n, iterations, lr, train_ep, sd):
    seed(sd)
    results = []
    policies = []
    print("---- Starting ----")
    for n in range(iterations):
        print("Iteration", n, end='\r')
        p1_a1, p2_a1, r1, r2 = lola_incomplete_simple_exp(dist_n, lr, train_ep)
        results.append((r1, r2))
        policies.append((p1_a1, p2_a1))
    print()

    ### From standard play
    x = [r1 for r1, r2 in results]
    y = [r2 for r1, r2 in results]
    plot_results(x, y, game='Incomp_four', save='incomplete_simple_lola', color='orange')
    save_results('Pfs_incomp_simp_lola', x, y)

    ### Cross play
    p1s = [a.tolist() for a, b in policies]
    p2s = [b.tolist() for a, b in policies]
    save_policies('Pols_incomp_simp_lola', p1s, p2s)

    save_results_and_policies('All_incomp_simp_lola', x, y, p1s, p2s)

def full_incomplete_simple_naive(dist_n, iterations, lr, train_ep, sd):
    seed(sd)
    results = []
    policies = []
    print("---- Starting ----")
    for n in range(iterations):
        print("Iteration", n, end='\r')
        p1_a1, p2_a1, r1, r2 = naive_incomplete_simple_exp(dist_n, lr, train_ep)
        results.append((r1, r2))
        policies.append((p1_a1, p2_a1))
    print()

    ### From standard play
    x = [r1 for r1, r2 in results]
    y = [r2 for r1, r2 in results]
    plot_results(x, y, game='Incomp_four', save='incomplete_simple_naive', color='orange')
    save_results('Pfs_incomp_simp_naive', x, y)

    ### Cross play
    p1s = [a.tolist() for a, b in policies]
    p2s = [b.tolist() for a, b in policies]
    save_policies('Pols_incomp_simp_naive', p1s, p2s)

    save_results_and_policies('All_incomp_simp_naive', x, y, p1s, p2s)

def experiment():
    seed
    do train/test loop, save results
    save results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-e', default='lola', choices=[
        'lola', 
        'incomplete_simple_lola',
        'incomplete_simple_naive'
    ])
    parser.add_argument('--iterations', '-i', default=40, type=int)
    parser.add_argument('--train_ep', '-te', default=100, type=int)
    parser.add_argument('--gamma', '-g', default=0.96, type=float)
    parser.add_argument('--learning_rate', '-lr', default=1, type=float)
    parser.add_argument('--seed', '-s', default=1234)
    parser.add_argument('--dist_n', '-d', default=0)
    args = parser.parse_args()

    seed(args.sd)
    if args.experiment == 'lola':
        full_lola(args.iterations, args.gamma, args.learning_rate, args.train_ep, args.seed)
    elif args.experiment == 'incomplete_simple_lola':
        full_incomplete_simple_lola(args.dist_n, args.iterations, args.learning_rate, args.train_ep, args.seed)
    elif args.experiment == 'incomplete_simple_naive':
        full_incomplete_simple_naive(args.dist_n, args.iterations, args.learning_rate, args.train_ep, args.seed)