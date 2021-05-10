import random
import argparse

import torch
import numpy as np

from utils import plot_results, save_results, seed, save_results_and_policies, save_plot
from test import several_test_exact, several_cross_test_exact
from games import get_game, ALL_GAMES
from train import train_policies
from train_steps import naive_step, lola_step
from quicksave import QuickSaver

step_funcs = {
    'lola': lola_step,
    'naive': naive_step,
}

def experiment(env, step_type, training_rounds, gamma, lr, train_ep, oneshot, test_ep, save_folder, config):
    step_func = step_funcs[step_type]
    # value_func = get_value_incomplete_oneshot if oneshot else get_value_incomplete_iterated

    print("---- Starting ----")
    p1s, p2s = train_policies(env, training_rounds, step_func, train_ep, gamma, lr) 
    r1s, r2s = several_test_exact(env, p1s, p2s)
    xr1s, xr2s = several_cross_test_exact(env, p1s, p2s, n_crosses=training_rounds)

    qs = QuickSaver(subfolder=save_folder)
    qs.save_json(config, name='config')

    ### Plot results
    plot_results(env, r1s, r2s, color='orange')
    plot_results(env, xr1s, xr2s, color='blue')
    save_plot(qs, 'results')

    ### Save results
    p1s = list(map(lambda p: p.tolist(), p1s))
    p2s = list(map(lambda p: p.tolist(), p2s))
    save_results_and_policies(qs, 'Pols_res', r1s, r2s, p1s, p2s)
    save_results(qs, 'XPfs', xr1s, xr2s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', type=str, default="")
    parser.add_argument('--oneshot', action='store_true')
    parser.add_argument('--step_type', '-st', default='naive', choices=[
        'naive',
        'lola'
    ])
    parser.add_argument('--game', default='IncompFour', choices=ALL_GAMES)
    parser.add_argument('--training_rounds', '-tr', default=40, type=int)
    parser.add_argument('--train_ep', '-te', default=100, type=int)
    parser.add_argument('--test_ep', '-tee', default=100, type=int)
    parser.add_argument('--gamma', '-g', default=0.96, type=float)
    parser.add_argument('--learning_rate', '-lr', default=1, type=float)
    parser.add_argument('--seed', '-s', default=1234)
    parser.add_argument('--prior_n', default=0, type=int)
    parser.add_argument('--p', default=0.7, type=int)
    parser.add_argument('--a', default=2, type=int)
    args = parser.parse_args()

    seed(args.seed)

    env = get_game(args.game, oneshot=args.oneshot, prior_n=args.prior_n, p=args.p, a=args.a)

    if args.save_folder == "":
        save_folder = "{}_{}".format(env.name, args.step_type)
    else:
        save_folder = args.save_folder

    experiment(env, args.step_type, args.training_rounds, args.gamma, args.learning_rate, 
        args.train_ep, args.oneshot, args.test_ep, save_folder, vars(args))
