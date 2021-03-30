import random

import numpy as np

def run_env(env, p1, p2):
    """
    Run env once, given mean reward per step
    """
    s, (t1, t2) = env.reset()
    s = np.argmax(s)
    p1, p2 = p1[t1], p2[t2]

    r1s, r2s = [], []
    done = False
    while not done:
        a1 = 0 if random.random() < p1[s] else 1
        a2 = 0 if random.random() < p2[s] else 1

        s, (r1, r2), done = env.step(a1, a2)
        s = np.argmax(s)
        r1s.append(r1)
        r2s.append(r2)

    return np.mean(r1s), np.mean(r2s)

def test(env, p1, p2, test_e):
    """
    Given two policies, test them test_e times and give the mean reward per step
    """
    all_r1s, all_r2s = [], []
    for _ in range(test_e):
        r1, r2 = run_env(env, p1, p2)
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