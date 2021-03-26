import random
import argparse

import torch
import numpy as np

from utils import plot_results, save_results, seed, save_results_and_policies, save_plot
from test_utils import test_iterated, test_cross_iterated, test_incomplete_oneshot, test_cross_incomplete_iterated
from games import get_game
from train_loop import train
from train_steps import naive_step, lola_step
from value_functions import get_value_incomplete_oneshot, get_value_iterated, get_value_incomplete_iterated
from quicksave import QuickSaver

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
    p1 = torch.randn(5, requires_grad=True)
    p2 = torch.randn(5, requires_grad=True)

    p1, p2 = train(env, p1, p2, get_value_iterated, step_func, train_ep, gamma, lr)

    # Convert to probabilities
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)

    r1, r2 = test_iterated(env, p1, p2)
    return p1, p2, r1, r2

def train_test_incomplete_iterated(env, step_func, gamma, lr, train_ep):
    p1 = torch.randn((2,5), requires_grad=True)
    p2 = torch.randn((2,5), requires_grad=True)

    p1, p2 = train(env, p1, p2, get_value_incomplete_iterated, step_func, train_ep, gamma, lr)

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
        'iterated': train_test_incomplete_iterated,
    },
}

step_funcs = {
    'lola': lola_step,
    'naive': naive_step,
}

cross_tests = {
    'complete': {
        'one_shot': None,
        'iterated': test_cross_iterated,
    },
    'incomplete': {
        'one_shot': None,
        'iterated': test_cross_incomplete_iterated,
    }, 
}

def experiment(game, game_type, info_type, step_type, iterations, gamma, lr, train_ep, dist_n, config):
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

    save_folder = "{}_{}_{}_{}".format(game, info_type, game_type, step_type)
    qs = QuickSaver(subfolder=save_folder)
    qs.save_json(config, name='config')

    ### From standard play
    x = [r1 for r1, r2 in results]
    y = [r2 for r1, r2 in results]
    plot_results(x, y, game=game, color='orange')

    ### Save policies
    p1s = [a.tolist() for a, b in policies]
    p2s = [b.tolist() for a, b in policies]

    ### Save all together
    save_results_and_policies(qs, 'Pols_res', x, y, p1s, p2s)
    
    ### Cross play
    cross_test = cross_tests[game_type][info_type]
    if cross_test is not None:
        r1_cross, r2_cross = cross_test(env, p1s, p2s, cross_tests=iterations)
        plot_results(r1_cross, r2_cross, game=game, color='blue')
        save_results(qs, 'XPfs', r1_cross, r2_cross)

    save_plot(qs, 'results')

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
    parser.add_argument('--dist_n', '-d', default=0, type=int)
    args = parser.parse_args()

    seed(args.seed)
    experiment(args.game,args.game_type, args.info_type, args.step_type, 
        args.iterations, args.gamma, args.learning_rate, args.train_ep, args.dist_n, vars(args))
