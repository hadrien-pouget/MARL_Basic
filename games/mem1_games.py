import random

import torch
import numpy as np

from games.base_games import IncompleteInfoGame
from test import test

pd_payoffs = [
    [(-1, -1), (-3, 0)], 
    [(0, -3),  (-2, -2)]
]

bos_payoffs = [
    [(4, 1), (0, 0)],
    [(0, 0), (2, 2)]
]

mp_payoffs = [
    [(2,0), (0,2)],
    [(0,2), (2,0)]
]

coord_payoffs = [
    [(2,2), (0,0)],
    [(0,0), (1,1)]
]

four_games = [[mp_payoffs, pd_payoffs], [coord_payoffs, bos_payoffs]]

class Mem1Game(IncompleteInfoGame):
    """ 
    Iterated game where state is the pair of actions taken in the previous game.
    Oneshot games are implemented as special case of this, where the game is automatically reset at each step.
    """
    def __init__(self, payoffs, prior_1=None, prior_2=None, prior_1_param=0, prior_2_param=0, oneshot=False):
        super().__init__(payoffs, prior_1=prior_1, prior_2=prior_2, prior_1_param=prior_1_param, prior_2_param=prior_2_param)
        self.n_obs = (self.action_space ** 2) + 1
        self.oneshot = oneshot
        self.steps_to_approx_iterated = 100 # number of steps taken in env to approximate an iterated game
        self.test_steps = 1 if self.oneshot else self.steps_to_approx_iterated # number of steps taken in env to approximate an iterated game

    def reset(self, prior=None):
        self.current_game_prior = self.prior_1 if prior is None else prior
        self.types = self.sample_types(prior=self.current_game_prior)        
        self.steps_taken = 0
        obs = np.zeros(self.n_obs)
        obs[-1] = 1
        obs = (obs, obs)
        return obs, self.types

    def step(self, a1, a2):
        self.steps_taken += 1
        obs = np.zeros(self.n_obs)
        obs[a1 * self.action_space + a2] = 1
        obs = (obs, obs)
        # (obs1, obs2), (r1, r2)
        result = obs, self.pfs[self.types[0]][self.types[1]][a1][a2]

        if self.oneshot:
            self.reset()

        return result

    def gen_rand_policies(self):
        """ 
        This should be passed through sigmoid to get a probability.
        This a bit awkward if you're doing oneshot - only the last index will be useful.
        """
        return torch.randn((self.n_types, self.n_obs), requires_grad=True), torch.randn((self.n_types, self.n_obs), requires_grad=True)

    def get_value(self, p1, p2, prior_1=None, prior_2=None, gamma=0.96, **kwargs):
        if prior_1 is None:
            prior_1 = self.prior_1

        if prior_2 is None:
            prior_2 = self.prior_2

        if self.oneshot:
            v1, _ = get_value_incomplete_oneshot(self.pfs, p1, p2, prior_1)
            _, v2 = get_value_incomplete_oneshot(self.pfs, p1, p2, prior_2)
        else:
            v1, _ = get_value_incomplete_iterated(self.pfs, p1, p2, prior_1, gamma)
            _, v2 = get_value_incomplete_iterated(self.pfs, p1, p2, prior_2, gamma)

        return v1, v2

    def play_game(self, p1, p2, prior=None):
        """
        Run env once, given mean reward per step
        """
        prior = self.prior_1 if prior is None else prior

        s, (t1, t2) = self.reset(prior=prior)
        s = np.argmax(s)
        p1, p2 = p1[t1], p2[t2]

        r1s, r2s = [], []
        for _ in range(self.test_steps):
            a1 = 0 if random.random() < p1[s] else 1
            a2 = 0 if random.random() < p2[s] else 1

            s, (r1, r2) = self.step(a1, a2)
            s = np.argmax(s)
            r1s.append(r1)
            r2s.append(r2)

        return np.mean(r1s), np.mean(r2s)

    def get_expected_step_payoffs(self, p1, p2, prior=None, **kwargs):
        prior = self.prior_1 if prior is None else prior
        if self.oneshot:
            return self.get_value(p1, p2, prior=prior)
        else:
            # For iterated games, do empirical value
            return test(self, prior, p1, p2, 100)

class IncompleteFour(Mem1Game):
    def __init__(self, prior_1_param=0, prior_2_param=0, oneshot=False):
        super().__init__(four_games, prior_1_param=prior_1_param, prior_2_param=prior_2_param, oneshot=oneshot)
        self.name = "IncompFour"

class DistInf(Mem1Game):
    def __init__(self, p, a, oneshot=False):
        """
        Implementing game from https://www.cambridge.org/core/journals/international-organization/article/abs/modeling-the-forms-of-international-cooperation-distribution-versus-information/F39FD45847A5593140E975E4EA2D580E
        without communication.
        p is the probability that they both prefer the same the outcome,
        a is the magnitude of preference for each outcome
        both players know p and a

        This game is more complicated, because the payoffs are tied to the prior through the probability p of being in a situation
        where we're cooperating. So defining two different priors (one for each agent) is hard.
        """
        bos = [
            [(a,1), (0,0)],
            [(0,0), (1,a)]
        ]

        prefA = [
            [(a,a), (0,0)],
            [(0,0), (1,1)]
        ]

        prefB = [
            [(1,1), (0,0)],
            [(0,0), (a,a)]
        ]

        # p(BoS | 1, 1) or p(BoS | 2, 2)
        bosp = (1-p)/(1+p)
        # p(prefA | 1, 1) or p(prefB | 2, 2)
        prefp = (2*p)/(1+p)
        # p(1, 1) or p(2, 2)
        oneonep = 0.25 * (1+p)
        # p(1, 2) or p(2, 1)
        onetwop = 0.25 * (1-p)

        oneone = [[(bos[a1][a2][0]*bosp + prefA[a1][a2][0]*prefp, bos[a1][a2][1]*bosp + prefA[a1][a2][1]*prefp) for a2 in [0, 1]] for a1 in [0, 1]]
        twotwo = [[(bos[a1][a2][0]*bosp + prefB[a1][a2][0]*prefp, bos[a1][a2][1]*bosp + prefB[a1][a2][1]*prefp) for a2 in [0, 1]] for a1 in [0, 1]]
        games = [[oneone, bos], [bos, twotwo]]
        prior = [[oneonep, onetwop], [onetwop, oneonep]]
        super().__init__(games, prior_1=prior, prior_2=prior, oneshot=oneshot)
        self.name = "DistInf"

def get_value_incomplete_oneshot(payoffs, p1, p2, dist):
    # Turn into probabilities
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)

    # Only the initial state matters
    p1 = p1[:,-1]
    p2 = p2[:,-1]

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

def get_value_incomplete_iterated(payoffs, p1, p2, dist, gamma):
    """
    p1[type][state]
    """
    vs_per_game_1 = []
    vs_per_game_2 = []
    for t1, t2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        v1, v2 = get_value_iterated(payoffs[t1][t2], p1[t1], p2[t2], None, gamma)
        vs_per_game_1.append(v1)
        vs_per_game_2.append(v2)

    vs_per_game_1 = torch.stack(vs_per_game_1)
    vs_per_game_2 = torch.stack(vs_per_game_2)
    dist = torch.tensor(dist).reshape(-1)

    v1 = torch.dot(vs_per_game_1, dist)
    v2 = torch.dot(vs_per_game_2, dist)

    return v1, v2

def get_value_iterated(payoffs, p1, p2, dist, gamma):
    # Turn into probabilities
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)
    # print("Probabilities are: \np1 {} \np2 {}".format(p1.detach(), p2.detach()))

    p1_sX_a1, p1_s0_a1 = p1.split([4, 1])
    p2_sX_a1, p2_s0_a1 = p2.split([4, 1])

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
    r1s = torch.tensor([r1 for a1 in payoffs for r1, r2 in a1], dtype=torch.float)
    r2s = torch.tensor([r2 for a1 in payoffs for r1, r2 in a1], dtype=torch.float)
    # payoffs for CC, CD, DC, DD

    inf_sum = torch.inverse(torch.eye(4) - (gamma * P))

    v1 = torch.dot(
        torch.mm(
            inf_sum, p0).reshape(-1),
        r1s
    )

    v2 = torch.dot(
        torch.mm(
            inf_sum, p0).reshape(-1),
        r2s
    )

    return v1, v2