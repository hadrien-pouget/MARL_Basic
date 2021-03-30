import random
import itertools

import numpy as np
from scipy.spatial import ConvexHull
import torch

def get_game(game, max_steps, dist_n):
    if game == 'IncompFour':
        return IncompleteFour(max_steps, dist_n)
    else:
        return None

class IncompleteInfoGame():
    """ 
    Implements an iterated game of incomplete information as an environment based on 
    a payoff matrix for two players
    
    Args:
        payoffs (list of list of list of list of tuples): payoffs of the game, 
    where the first index is player 1's type, the second is player 2's type,
    the third index is player 1's action, and the fourth is player 2's action. Currently
    assume they have the same actions and types available
        max_steps (int)
        action_names (optional list of str): for clarity when printing payoffs
    """
    def __init__(self, games, max_steps, dist_n=0):
        self.pfs = games
        self.n_types = len(games)
        self.n_games = self.n_types ** 2
        self.action_space = len(self.pfs[0][0])
        self.n_obs = (self.action_space ** 2) + 1

        self.preset_dists = [
            [[1/self.n_games for _ in range(self.n_types)] for _ in range(self.n_types)],
            # The following are for games with two types
            [[1., 0.], [0., 0.]],
            [[0., 1.], [0., 0.]],
            [[0., 0.], [1., 0.]],
            [[0., 0.], [0., 1.]],
            [[0.1, 0.4], [0.2, 0.3]],
            [[0.02, 0.94], [0.02, 0.02]],
        ]
        self.dist = self.preset_dists[dist_n]

        self.max_steps = max_steps
        self.types = None
        self.steps_taken = 0

    def _sample_types(self):
        tot = 0
        t1 = -1
        r = random.random()
        done = False
        while not done:
            t1 += 1
            games = self.dist[t1]
            t2 = -1
            for g_prob in games:
                t2 += 1
                tot += g_prob
                if tot >= r:
                    done = True
                    break

        return t1, t2

    def reset(self):
        self.types = self._sample_types()
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
        # (obs1, obs2), (r1, r2), done 
        return obs, self.pfs[self.types[0]][self.types[1]][a1][a2], self.steps_taken >= self.max_steps

    def normal_form(self):
        actions = [0, 1]
        rep_actions = [actions] * (2 * self.n_types)
        # a12 is action of player 1 for type 2
        rs = []
        for strats in itertools.product(*rep_actions):
            r1, r2 = 0, 0
            for t1, t2 in itertools.product(range(self.n_types), range(self.n_types)):
                r1 += self.dist[t1][t2] * self.pfs[t1][t2][strats[t1]][strats[self.n_types + t2]][0]
                r2 += self.dist[t1][t2] * self.pfs[t1][t2][strats[t1]][strats[self.n_types + t2]][1]
            rs.append((r1, r2))
        
        rowlen = len(actions)**self.n_types
        payoffs = [[rs[(p1 * rowlen)+p2] for p2 in range(rowlen)] for p1 in range(rowlen)]
        return payoffs

    def print_normalform(self):
        print("\n".join(map(str, self.normal_form())))
    
    def get_pure_outcomes_as_points(self):
        nform = self.normal_form()
        points = [rs for strat1 in nform for rs in strat1]
        return points

    def outcomes_polygon(self):
        points = self.get_pure_outcomes_as_points()
        hull = ConvexHull(points)
        polygon_xs = [points[p][0] for p in hull.vertices]
        polygon_ys = [points[p][1] for p in hull.vertices]
        return polygon_xs, polygon_ys

    def gen_rand_policy(self):
        """ 
        This should be passed through sigmoid to get a probability
        """
        return torch.randn((self.n_types, self.n_obs), requires_grad=True)

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

class IncompleteFour(IncompleteInfoGame):
    def __init__(self, max_steps=100, dist_n=0):
        super().__init__(four_games, max_steps, dist_n=dist_n)
