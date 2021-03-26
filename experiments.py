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

    p1 = torch.randn(2, requires_grad=True)
    p2 = torch.randn(2, requires_grad=True)

    p1, p2 = train(env, p1, p2, get_value_incomplete_oneshot, step_func, train_ep, gamma, lr)

    # Convert to probabilities
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)

    r1, r2 = incomplete_oneshot_test(env, p1, p2, test_e=100)
    return p1, p2, r1, r2

def train_test_iterated(env, step_func, gamma, lr, train_ep):
    """
    Given a game, initialise random agents, train with lola,
    and return average payoff for each step
    """
    p1 = torch.randn(5, requires_grad=True, dtype=torch.float64)
    p2 = torch.randn(5, requires_grad=True, dtype=torch.float64)

    p1, p2 = train(env, p1, p2, get_value_iterated, step_func, train_ep, gamma, lr)

    # Convert to probabilities
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)

    r1, r2 = test_iterated(env, p1, p2)
    return p1, p2, r1, r2

test_trains = {
    'complete': {
        'one_shot': None,
        'iterated': train_test_iterated,
    },
    'incomplete': {
        'one_shot': train_test_incomplete_oneshot,
        'iterated': None,
    },
}

step_funcs = {
    'lola': lola_step,
    'naive': naive_step,
}

def experiment(game, game_type, info_type, step_type, iterations, gamma, lr, train_ep, dist_n=None):
    results = []
    policies = []
    env = get_game(game, dist_n=dist_n)
    test_train = test_trains[info_type][game_type]
    step_func = step_funcs[step_type]
    print("---- Starting ----")
    for n in range(iterations):
        print("Iteration", n, end='\r')
        p1_a1, p2_a1, r1, r2 = test_train(env, step_func, gamma, lr, train_ep)
        results.append((r1, r2))
        policies.append((p1_a1, p2_a1))
    print()

    save_pstfix = "_{}_{}_{}_{}".format(game, info_type, game_type, step_type)

    ### From standard play
    x = [r1 for r1, r2 in results]
    y = [r2 for r1, r2 in results]
    plot_results(x, y, game=game, save=save_pstfix, color='orange')
    save_results('Pfs' + save_pstfix, x, y)

    ### Save policies
    p1s = [a.tolist() for a, b in policies]
    p2s = [b.tolist() for a, b in policies]
    save_policies('Pols' + save_pstfix, p1s, p2s)

    ### Save all together
    save_results_and_policies('All' + save_pstfix, x, y, p1s, p2s)
    
    ### Cross play
    if game_type == 'iterated':
        r1_cross, r2_cross = test_cross_iterated(env, p1s, p2s)
        plot_results(r1_cross, r2_cross, game=game, save=save_pstfix, color='blue')
        save_results('XPfs' + save_pstfix, r1_cross, r2_cross)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_type', '-gt', default='iterated', choices=[
        'iterated',
        'one_shot'
    ])
    parser.add_argument('--info_type', '-it', default='complete', choices=[
        'complete',
        'incomplete'
    ])
    parser.add_argument('--step_type', '-st', default='naive', choices=[
        'naive',
        'lola'
    ])
    parser.add_argument('--game', default='PD', choices=[
        'PD',
        'BoS',
        'Incomp_four'
    ])
    parser.add_argument('--iterations', '-i', default=40, type=int)
    parser.add_argument('--train_ep', '-te', default=100, type=int)
    parser.add_argument('--gamma', '-g', default=0.96, type=float)
    parser.add_argument('--learning_rate', '-lr', default=1, type=float)
    parser.add_argument('--seed', '-s', default=1234)
    parser.add_argument('--dist_n', '-d', default=0)
    args = parser.parse_args()

    seed(args.seed)
    experiment(args.game,args.game_type, args.info_type, args.step_type, 
        args.iterations, args.gamma, args.learning_rate, args.train_ep, args.dist_n)
