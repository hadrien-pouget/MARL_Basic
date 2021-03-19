import random

import torch
import numpy as np

from games import PD, BoS
from utils import zero_grad

def get_values(payoffs, p1_a1, p2_a1, gamma):
    # Turn into probabilities
    p1_a1 = torch.sigmoid(p1_a1)
    p2_a1 = torch.sigmoid(p2_a1)
    # print("Probabilities are: \np1 {} \np2 {}".format(p1_a1.detach(), p2_a1.detach()))

    p1_sX_a1, p1_s0_a1 = p1_a1.split([4, 1])
    p2_sX_a1, p2_s0_a1 = p2_a1.split([4, 1])

    ### Get p0
    p1_s0_a12 = torch.stack([p1_s0_a1, p1_s0_a1, 1 - p1_s0_a1, 1 - p1_s0_a1])
    p2_s0_a12 = torch.stack([p2_s0_a1, 1 - p2_s0_a1, p2_s0_a1, 1 - p2_s0_a1])
    p0 = p1_s0_a12 * p2_s0_a12
    # size: [4,] probability of [CC, CD, DC, DD] from s0

    ### Get transition matrix
    P = torch.stack([
        torch.mul(p1_sX_a1, p2_sX_a1),
        torch.mul(p1_sX_a1, 1-p2_sX_a1),
        torch.mul(1-p1_sX_a1, p2_sX_a1),
        torch.mul(1-p1_sX_a1, 1-p2_sX_a1)
    ])
    P.t()
    # size: [4,4] probability of transition - first ind is s', second is s, 

    ### Calculate V1 and V2
    r1 = torch.tensor([p1 for a1 in payoffs for p1, p2 in a1], dtype=torch.float64)
    r2 = torch.tensor([p2 for a1 in payoffs for p1, p2 in a1], dtype=torch.float64)
    # payoffs for CC, CD, DC, DD

    inf_sum = torch.inverse(torch.eye(4) - (gamma * P))

    v1 = torch.dot(
        torch.mm(
            inf_sum, p0).reshape(-1),
        r1
    )

    v2 = torch.dot(
        torch.mm(
            inf_sum, p0).reshape(-1),
        r2
    )

    return v1, v2

def lola_train(env, p1_a1, p2_a1, episodes, gamma, lr, verbose=0):
    """ 
    Given initial parameters p1_a1, p2_a1 (pre-sigmoid, not probabilities),
    train according to exact lola
    """
    if verbose > 0:
        print()
        print("---- Beginning training ----")
    for e in range(episodes):
        v1, v2 = get_values(env.pfs, p1_a1, p2_a1, gamma)
        if verbose > 0:
            print("E {:2} | v1: {:0.2f} | v2: {:0.2f}".format(e+1, v1.item(), v2.item()))

        zero_grad(p1_a1)
        zero_grad(p2_a1)

        ### Gradients of each value function w/ respect to each set of parameters
        [v1_grad_p1, v1_grad_p2] = torch.autograd.grad(v1, [p1_a1, p2_a1], create_graph=True, only_inputs=True)
        [v2_grad_p1, v2_grad_p2] = torch.autograd.grad(v2, [p1_a1, p2_a1], create_graph=True, only_inputs=True)

        ### p1 lola gradient
        multiply = torch.dot(v2_grad_p2, v1_grad_p2)
        v1_approx = v1 + multiply
        [p1_grad] = torch.autograd.grad(v1_approx, p1_a1, retain_graph=True, only_inputs=True)

        ### p2 lola gradient
        multiply = torch.dot(v1_grad_p1, v2_grad_p1)
        v2_approx = v2 + multiply
        [p2_grad] = torch.autograd.grad(v2_approx, p2_a1, retain_graph=True, only_inputs=True)

        ### Take gradient steps
        with torch.no_grad():
            p1_a1 += p1_grad * lr
            p2_a1 += p2_grad * lr

    if verbose > 0:
        v1, v2 = get_values(env.pfs, p1_a1, p2_a1, gamma)
        print("Final Values are: p1 {:0.2f} | p2 {:0.2f}".format(v1.item(), v2.item()))
    return p1_a1.detach().clone(), p2_a1.detach().clone()
