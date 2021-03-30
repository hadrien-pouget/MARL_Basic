import random
import argparse

import torch
import numpy as np

from utils import plot_results, save_results, seed, save_results_and_policies, save_plot
from test import several_test, several_cross_test
from games import get_game
from train import train_policies
from train_steps import naive_step, lola_step
from value_functions import get_value_incomplete_oneshot, get_value_incomplete_iterated
from quicksave import QuickSaver

step_funcs = {
    'lola': lola_step,
    'naive': naive_step,
}

def experiment(game, step_type, training_rounds, gamma, lr, train_ep, dist_n, oneshot, test_ep, config):
    max_steps = 1 if oneshot else 100
    env = get_game(game, max_steps, dist_n)
    step_func = step_funcs[step_type]
    value_func = get_value_incomplete_oneshot if oneshot else get_value_incomplete_iterated

    print("---- Starting ----")
    p1s, p2s = train_policies(env, training_rounds, value_func, step_func, train_ep, gamma, lr) 
    r1s, r2s = several_test(env, p1s, p2s, test_ep)
    xr1s, xr2s = several_cross_test(env, p1s, p2s, test_ep, n_crosses=training_rounds)

    save_folder = "{}_{}_{}".format(game, step_type, "oneshot" if oneshot else "iterated")
    qs = QuickSaver(subfolder=save_folder)
    qs.save_json(config, name='config')

    ### Plot results
    plot_results(env, r1s, r2s, game=game, color='orange')
    plot_results(env, xr1s, xr2s, game=game, color='blue')
    save_plot(qs, 'results')

    ### Save results
    p1s = list(map(lambda p: p.tolist(), p1s))
    p2s = list(map(lambda p: p.tolist(), p2s))
    save_results_and_policies(qs, 'Pols_res', r1s, r2s, p1s, p2s)
    save_results(qs, 'XPfs', xr1s, xr2s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oneshot', action='store_true')
    parser.add_argument('--step_type', '-st', default='naive', choices=[
        'naive',
        'lola'
    ])
    parser.add_argument('--game', default='IncompFour', choices=[
        'IncompFour'
    ])
    parser.add_argument('--training_rounds', '-tr', default=40, type=int)
    parser.add_argument('--train_ep', '-te', default=100, type=int)
    parser.add_argument('--test_ep', '-tee', default=100, type=int)
    parser.add_argument('--gamma', '-g', default=0.96, type=float)
    parser.add_argument('--learning_rate', '-lr', default=1, type=float)
    parser.add_argument('--seed', '-s', default=1234)
    parser.add_argument('--dist_n', '-d', default=0, type=int)
    args = parser.parse_args()

    seed(args.seed)
    experiment(args.game, args.step_type, args.training_rounds, args.gamma, args.learning_rate, 
        args.train_ep, args.dist_n, args.oneshot, args.test_ep, vars(args))
