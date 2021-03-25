import random

import numpy as np

def test(env, p1_a1, p2_a1, test_e=1):
    """
    Average rewards for each step, for p1 and p2, tested over test_e episodes
    p1_a1 and p2_a1 should be probabilities (other parts of the code use them pre-sigmoid)

    Not much of a point in doing it over several episodes at the moment
    """
    all_r1s, all_r2s = [], []
    for _ in range(test_e):
        s = np.argmax(env.reset())
        r1s = []
        r2s = []
        done = False
        while not done:
            a1 = 0 if random.random() < p1_a1[s] else 1
            a2 = 0 if random.random() < p2_a1[s] else 1

            s, (r1, r2), done = env.step(a1, a2)
            s = np.argmax(s)
            r1s.append(r1)
            r2s.append(r2)

        all_r1s.append(np.mean(r1s))
        all_r2s.append(np.mean(r2s))
    
    return np.mean(all_r1s), np.mean(all_r2s)

def test_cross(env, p1s, p2s, test_e=1, cross_tests=40):
    all_r1s = []
    all_r2s = []
    for _ in range(cross_tests):
        p1_a1 = random.choice(p1s)
        p2_a1 = random.choice(p2s)

        r1, r2 = test(env, p1_a1, p2_a1, test_e=test_e)
        all_r1s.append(r1)
        all_r2s.append(r2)
    return all_r1s, all_r2s

def incomplete_simple_test(env, p1, p2, test_e=1):
    all_r1s, all_r2s = [], []
    games = [0,0,0,0] # For debugging purposes
    for _ in range(test_e):
        t1, t2 = env.reset()
        games[(2*t1)+t2] += 1
        a1 = 0 if random.random() < p1[t1] else 1
        a2 = 0 if random.random() < p2[t2] else 1  
        r1, r2 = env.step(a1, a2)     
        all_r1s.append(r1) 
        all_r2s.append(r2)

    return np.mean(all_r1s), np.mean(all_r2s)
