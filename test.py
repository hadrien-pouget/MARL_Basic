import random

import numpy as np
import torch

### Exact value calculation
def several_test_exact(env, prior, p1s, p2s):
    all_r1s, all_r2s = [], []
    for p1, p2 in zip(p1s, p2s):
        r1, r2 = env.get_expected_step_payoffs(p1, p2, prior)
        all_r1s.append(r1.item())
        all_r2s.append(r2.item())
    return all_r1s, all_r2s

def choose_policies(p1s, p2s):
    p1 = random.choice(p1s)
    p2 = random.choice(p2s)
    return p1, p2

def several_cross_test_exact(env, prior, p1s, p2s, n_crosses):
    """
    Run tests for many combinations of policies
    """
    all_r1s, all_r2s = [], []
    for _ in range(n_crosses):
        p1, p2 = choose_policies(p1s, p2s)
        r1, r2 = env.get_expected_step_payoffs(p1, p2, prior)
        all_r1s.append(r1.item())
        all_r2s.append(r2.item())
    return all_r1s, all_r2s

# ### Empirical testing
# def test(env, p1, p2, test_e):
#     """
#     Given two policies, test them test_e times and give the mean reward per step
#     """
#     all_r1s, all_r2s = [], []
#     for _ in range(test_e):
#         r1, r2 = env.play_game(p1, p2)
#         all_r1s.append(r1)
#         all_r2s.append(r2)

#     return np.mean(all_r1s), np.mean(all_r2s)

# def several_test(env, p1s, p2s, test_e):
#     """
#     Do the above for many policies
#     """
#     all_r1s, all_r2s = [], []
#     for n, (p1, p2) in enumerate(zip(p1s, p2s)):
#         print("Testing round:", n, end='\r')
#         r1, r2 = test(env, p1, p2, test_e)
#         all_r1s.append(r1)
#         all_r2s.append(r2)
#     print()
#     return all_r1s, all_r2s

# def cross_test(env, p1s, p2s, test_e):
#     """
#     Run a test with policies that weren't trained together
#     """
#     p1, p2 = choose_policies(p1s, p2s)
#     return test(env, p1, p2, test_e)

# def several_cross_test(env, p1s, p2s, test_e, n_crosses):
#     """
#     Run tests for many combinations of policies
#     """
#     all_r1s, all_r2s = [], []
#     for n in range(n_crosses):
#         print("Cross Testing round:", n, end='\r')
#         r1, r2 = cross_test(env, p1s, p2s, test_e)
#         all_r1s.append(r1)
#         all_r2s.append(r2)
#     print()
#     return all_r1s, all_r2s
