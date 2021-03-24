import random
import itertools

import numpy as np

class IteratedGame():
    """ Implements an iterated game as an environment based on 
    a payoff matrix for two players
    
    Args:
        payoffs (list of list of tuples): payoffs of the game, 
    where the first index is player 1, and the second is player 2. Currently
    assume they have the same actions available
        max_steps (int)
        action_names (optional list of str): for clarity when printing payoffs
    """

    def __init__(self, payoffs, max_steps, action_names=None):
        self.pfs = payoffs
        self.max_steps = max_steps
        self.action_space = (len(self.pfs), len(self.pfs[0]))
        self.n_obs = self.action_space[0] * self.action_space[1]
        self.n_obs += 1 # Initial state 
        self.steps_taken = 0

        if action_names is not None:
            self.action_names = action_names
        else:
            self.action_names == list(map(str, (range(len(self.pfs)))))

    def __str__(self):
        max_name = max([len(n) for n in self.action_names])

        top = "\n" + " " * max_name + "|"
        for n in self.action_names:
            top += n + "|"

        bar = "\n" + "-" * (len(top) - 1)

        mid = ""
        for i, n1 in enumerate(self.action_names):
            mid += "\n" + n1 + " " * (max_name - len(n1)) + "|"
            for j, n2 in enumerate(self.action_names):
                scores = str(self.pfs[i][j][0]) + ", " + str(self.pfs[i][j][1])
                mid += scores + " " * (len(n2) - len(scores)) + "|"

        return top + bar + mid

    def step(self, a1, a2):
        self.steps_taken += 1
        obs = np.zeros(self.n_obs)
        obs[a1 * self.action_space[1] + a2] = 1
        obs = (obs, obs)
        # (obs1, obs2), (r1, r2), done 
        return obs, self.pfs[a1][a2], self.steps_taken >= self.max_steps

    def reset(self):
        self.steps_taken = 0
        obs = np.zeros(self.n_obs)
        obs[-1] = 1
        obs = (obs, obs)
        return obs
        

pd_payoffs = [
    [(-1, -1), (-3, 0)], 
    [(0, -3),  (-2, -2)]
]
pd_actions = ["cooperate", "defect"]

class PD(IteratedGame):
    def __init__(self):
        super().__init__(pd_payoffs, 100, action_names=pd_actions)

bos_payoffs = [
    [(4, 1), (0, 0)],
    [(0, 0), (2, 2)]
]
bos_actions = ["Bach", "Stravinsky"]

class BoS(IteratedGame):
    def __init__(self):
        super().__init__(bos_payoffs, 100, action_names=bos_actions)

mp_payoffs = [
    [(2,0), (0,2)],
    [(0,2), (2,0)]
]

class MP(IteratedGame):
    def __init__(self):
        super().__init__(mp_payoffs, 100)

coord_payoffs = [
    [(2,2), (0,0)],
    [(0,0), (1,1)]
]

class Coord(IteratedGame):
    def __init__(self):
        super().__init__(coord_payoffs, 100)

def get_game(game):
    if game == 'PD':
        return PD
    elif game == 'BoS':
        return BoS

class IncompleteInfoGame():
    """
    A one step game (although can be reset to resample types)

    For types T, games should be a |T|x|T| matrix of 
    games payoffs. First ind is player 1's type, second is
    player 2's
    """
    def __init__(self, games, dist='unif'):
        self.pfs = games
        self.n_types = len(games)
        self.n_games = self.n_types ** 2
        self.dist = dist
        if self.dist == 'unif':
            self.dist = [[1/self.n_games for _ in range(self.n_types)] for _ in range(self.n_types)]
        self.types = None

    def _sample_types(self):
        """
        Uniform, or supply joint prob
        """
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
        return self.types[0], self.types[1]

    def step(self, a1, a2):
        return self.pfs[self.types[0]][self.types[1]][a1][a2]

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

four_games = [[mp_payoffs, pd_payoffs], [coord_payoffs, bos_payoffs]]

class IncompleteFour(IncompleteInfoGame):
    def __init__(self, dist_n=0):
        self.preset_dists = [
            'unif',
            [[1., 0.], [0., 0.]],
            [[0., 1.], [0., 0.]],
            [[0., 0.], [1., 0.]],
            [[0., 0.], [0., 1.]],
            [[0.1, 0.4], [0.2, 0.3]]
        ]
        super().__init__(four_games, dist=self.preset_dists[dist_n])
