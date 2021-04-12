import random

import numpy as np

def test(env, p1, p2, test_e):
    """
    Given two policies, test them test_e times and give the mean reward per step
    """
    all_r1s, all_r2s = [], []
    for _ in range(test_e):
        r1, r2 = env.play_game(p1, p2)
        all_r1s.append(r1)
        all_r2s.append(r2)

    return np.mean(all_r1s), np.mean(all_r2s)

def several_test(env, p1s, p2s, test_e):
    """
    Do the above for many policies
    """
    all_r1s, all_r2s = [], []
    for n, (p1, p2) in enumerate(zip(p1s, p2s)):
        print("Testing round:", n, end='\r')
        r1, r2 = test(env, p1, p2, test_e)
        all_r1s.append(r1)
        all_r2s.append(r2)
    print()
    return all_r1s, all_r2s

def choose_policies(p1s, p2s):
    p1 = random.choice(p1s)
    p2 = random.choice(p2s)
    return p1, p2

def cross_test(env, p1s, p2s, test_e):
    """
    Run a test with policies that weren't trained together
    """
    p1, p2 = choose_policies(p1s, p2s)
    return test(env, p1, p2, test_e)

def several_cross_test(env, p1s, p2s, test_e, n_crosses):
    """
    Run tests for many combinations of policies
    """
    all_r1s, all_r2s = [], []
    for n in range(n_crosses):
        print("Cross Testing round:", n, end='\r')
        r1, r2 = cross_test(env, p1s, p2s, test_e)
        all_r1s.append(r1)
        all_r2s.append(r2)
    print()
    return all_r1s, all_r2s
