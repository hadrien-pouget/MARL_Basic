import random
import argparse

import torch
import numpy as np

from utils import save_results, seed, save_plot
from plot_utils import plot_results
from test import several_test, several_cross_test
from games import get_game, ALL_GAMES
from train import train_policies
from train_steps import naive_step, lola_step
from quicksave import QuickSaver

step_funcs = {
    'lola': lola_step,
    'naive': naive_step,
}

def experiment(env, step_type, training_rounds, gamma, lr, train_ep, oneshot, test_ep, save_folder, device, config):
    """
    Run an experiment where agents are trained and tested (including cross-play)

    training_rounds is the number of pairs of policies trained
    """
    step_func = step_funcs[step_type]

    print("---- Starting ----")
    ### Train policies
    p1s, p2s = train_policies(env, training_rounds, step_func, train_ep, gamma, lr, device) 

    qs = QuickSaver(subfolder=save_folder)
    qs.save_json(config, name='config')

    ### Save policies
    list_p1s = list(map(lambda p: p.tolist(), p1s))
    list_p2s = list(map(lambda p: p.tolist(), p2s))
    save_results(qs, 'Pols', list_p1s, list_p2s)

    for name, prior in env.generate_test_priors():
        print("Testing prior", name, "...")
        ### Test policies
        r1s, r2s = several_test(env, prior, p1s, p2s)
        xr1s, xr2s = several_cross_test(env, prior, p1s, p2s, n_crosses=training_rounds)
    
        ### Plot results
        plot_results(env, prior, r1s, r2s, color='orange')
        plot_results(env, prior, xr1s, xr2s, color='blue')
        save_plot(qs, name + '_results')

        ### Save results
        save_results(qs, name + '_Pfs', r1s, r2s)
        save_results(qs, name + '_XPfs', xr1s, xr2s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--save_folder', type=str, default="")
    parser.add_argument('--game', default='IncompFour', choices=ALL_GAMES)

    # Housekeeping
    parser.add_argument('--seed', '-s', default=1234)
    parser.add_argument('--force_cpu', action='store_true')

    # For making the prior
    parser.add_argument('--prior_1_param', nargs='*', default=[0])
    parser.add_argument('--prior_2_param', nargs='*', default=[0])

    # For iterated games (or forcing an iterated game to be oneshot)
    parser.add_argument('--gamma', '-g', default=0.96, type=float)
    parser.add_argument('--oneshot', action='store_true')

    # For communication game
    parser.add_argument('--a', default=2, type=int,
        help="Intensity of prefernces in distvinf games")
    parser.add_argument('--p', default=0.5, type=int, 
        help="Used for distvinf game, where the prior_param is ignored.")

    # For training and testing
    parser.add_argument('--training_rounds', '-tr', default=40, type=int)
    parser.add_argument('--train_ep', '-te', default=100, type=int)
    parser.add_argument('--learning_rate', '-lr', default=1, type=float)
    parser.add_argument('--step_type', '-st', default='naive', choices=[
        'naive',
        'lola'
    ])
    parser.add_argument('--test_ep', '-tee', default=100, type=int)

    # Parse
    args = parser.parse_args()

    # Process
    seed(args.seed)

    env = get_game(**vars(args))

    if args.save_folder == "":
        save_folder = "{}_{}".format(env.name, args.step_type)
    else:
        save_folder = args.save_folder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu' if args.force_cpu else device

    experiment(env, args.step_type, args.training_rounds, args.gamma, args.learning_rate, 
        args.train_ep, args.oneshot, args.test_ep, save_folder, device, vars(args))
