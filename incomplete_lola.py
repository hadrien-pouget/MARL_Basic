import random

import torch
import numpy as np

from games import PD, BoS
from utils import zero_grad

def incomplete_simple_get_values(payoffs, dist, p1, p2):
    # Turn into probabilities
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)

    p11, p12 = p1.split([1,1])
    # Player 1, types 1 and 2
    x2_p11 = torch.cat([p11, p11])
    x2_np11 = torch.cat([1-p11, 1-p11])
    x2_p12 = torch.cat([p12, p12])
    x2_np12 = torch.cat([1-p12, 1-p12])
    temp_p11 = torch.cat([x2_p11, x2_np11, x2_p11, x2_np11])
    temp_p12 = torch.cat([x2_p12, x2_np12, x2_p12, x2_np12])
    long_p1 = torch.cat([temp_p11, temp_p12])

    p21, p22 = p2.split([1,1])
    p21_np21 = torch.cat([p21, 1-p21])
    p22_np22 = torch.cat([p22, 1-p22])
    temp_p21 = torch.cat([p21_np21, p21_np21])
    temp_p22 = torch.cat([p22_np22, p22_np22])
    long_p2 = torch.cat([temp_p21, temp_p22, temp_p21, temp_p22])

    p_outcomes = (long_p1 * long_p2).reshape((4,4))
    # Probabilities. First dim is types (11, 12, 21, 22) second is outcome (UL, UR, DL, DR)

    r1 = torch.tensor([[p1 for a1 in game for p1, p2 in a1] for t1 in payoffs for game in t1], dtype=torch.float).t()
    r2 = torch.tensor([[p2 for a1 in game for p1, p2 in a1] for t1 in payoffs for game in t1], dtype=torch.float).t()
    # Rewards. First dim is outcome, second is types

    type_payoffs1 = torch.mm(p_outcomes, r1).diagonal()
    type_payoffs2 = torch.mm(p_outcomes, r2).diagonal()
    # vector of expected payoff for types 11 12 21 22

    dist = torch.tensor(dist).reshape(-1)
    # vector of probabilities of 11 12 21 22

    v1 = torch.dot(type_payoffs1, dist)
    v2 = torch.dot(type_payoffs2, dist)

    return v1, v2

def incomplete_simple_naive_train(env, p1, p2, episodes, lr, verbose=0):
    if verbose > 1:
        print()
        print("---- Beginning training ----")
    for e in range(episodes):
        v1, v2 = incomplete_simple_get_values(env.pfs, env.dist, p1, p2)
        if verbose > 1:
            with torch.no_grad():
                print("E {:3} | v1: {:0.2f} | v2: {:0.2f} | p1: {:0.2f} {:0.2f} | p2: {:0.2f} {:0.2f}".format(e+1, v1.item(), v2.item(), *torch.sigmoid(p1).tolist(), *torch.sigmoid(p2).tolist()))
                # print("E {:2} | p1: {:0.2f} {:0.2f} | p2: {:0.2f} {:0.2f}".format(e+1, *p1.tolist(), *p2.tolist()))

        zero_grad(p1)
        zero_grad(p2)

        ### Gradients of each value function w/ respect to each set of parameters
        [p1_grad] = torch.autograd.grad(v1, [p1], create_graph=True, only_inputs=True)
        [p2_grad] = torch.autograd.grad(v2, [p2], create_graph=True, only_inputs=True)

        ### Take gradient steps
        with torch.no_grad():
            p1 += p1_grad * lr
            p2 += p2_grad * lr

    if verbose > 0:
        with torch.no_grad():
            v1, v2 = incomplete_simple_get_values(env.pfs, env.dist, p1, p2)
            print("E   F | v1: {:0.2f} | v2: {:0.2f} | p1: {:0.2f} {:0.2f} | p2: {:0.2f} {:0.2f}".format(v1.item(), v2.item(), *torch.sigmoid(p1).tolist(), *torch.sigmoid(p2).tolist()))
    return p1.detach().clone(), p2.detach().clone() 

def incomplete_simple_lola_train(env, p1, p2, episodes, lr, verbose=0):
    """ 
    For a one-shot incomplete information game. p1 and p2 are a list 
    of numbers representing the pre-sigmoid probability for action 0.
    """
    if verbose > 1:
        print()
        print("---- Beginning training ----")
    for e in range(episodes):
        v1, v2 = incomplete_simple_get_values(env.pfs, env.dist, p1, p2)
        if verbose > 1:
            with torch.no_grad():
                print("E {:3} | v1: {:0.2f} | v2: {:0.2f} | p1: {:0.2f} {:0.2f} | p2: {:0.2f} {:0.2f}".format(e+1, v1.item(), v2.item(), *torch.sigmoid(p1).tolist(), *torch.sigmoid(p2).tolist()))
                # print("E {:2} | p1: {:0.2f} {:0.2f} | p2: {:0.2f} {:0.2f}".format(e+1, *p1.tolist(), *p2.tolist()))

        zero_grad(p1)
        zero_grad(p2)

        ### Gradients of each value function w/ respect to each set of parameters
        [v1_grad_p1, v1_grad_p2] = torch.autograd.grad(v1, [p1, p2], create_graph=True, only_inputs=True)
        [v2_grad_p1, v2_grad_p2] = torch.autograd.grad(v2, [p1, p2], create_graph=True, only_inputs=True)

        ### p1 lola gradient
        multiply = torch.dot(v2_grad_p2, v1_grad_p2)
        v1_approx = v1 + multiply
        [p1_grad] = torch.autograd.grad(v1_approx, p1, retain_graph=True, only_inputs=True)

        ### p2 lola gradient
        multiply = torch.dot(v1_grad_p1, v2_grad_p1)
        v2_approx = v2 + multiply
        [p2_grad] = torch.autograd.grad(v2_approx, p2, retain_graph=True, only_inputs=True)

        ### Take gradient steps
        with torch.no_grad():
            p1 += p1_grad * lr
            p2 += p2_grad * lr

    if verbose > 0:
        with torch.no_grad():
            v1, v2 = incomplete_simple_get_values(env.pfs, env.dist, p1, p2)
            print("E   F | v1: {:0.2f} | v2: {:0.2f} | p1: {:0.2f} {:0.2f} | p2: {:0.2f} {:0.2f}".format(v1.item(), v2.item(), *torch.sigmoid(p1).tolist(), *torch.sigmoid(p2).tolist()))
    return p1.detach().clone(), p2.detach().clone()
