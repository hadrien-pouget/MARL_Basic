import random
import itertools
from abc import ABC, abstractmethod

import numpy as np
import scipy
from scipy.spatial import ConvexHull

class IncompleteInfoGame(ABC):
    """
    This includes none of the functionality of actually playing the game, which is 
    left abstract. It allows basic 'viewing' functionality for incomplete information
    games, that remains constant across the different types of games. It also sets up 
    some useful parameters.

    Assumes there are only two players.
    """ 
    def __init__(self, payoffs, prior_1=None, prior_2=None, prior_1_param=0, prior_2_param=0):
        self.pfs = payoffs
        self.n_types = len(payoffs)
        self.n_games = self.n_types ** 2
        self.action_space = len(self.pfs[0][0])

        if prior_1 is None:
            self.prior_1 = self.select_prior(prior_1_param)
        else:
            self.prior_1 = prior_1
        if prior_2 is None:
            self.prior_2 = self.select_prior(prior_2_param)
        else:
            self.prior_2 = prior_2

    @abstractmethod
    def play_game(self, p1, p2):
        """ play game, return payoffs"""
        pass

    @abstractmethod
    def gen_rand_policies(self):
        """
        Return random policies p1, p2 for playing the game.
        """
        pass

    @abstractmethod
    def get_value(self, p1, p2, prior_1=None, prior_2=None, **kwargs):
        """
        Get v1, v2 if played using p1 and p2, each according to their own prior.
        Should use self.prior_1 for v1 and self.prior_2 for v2 if None is given.
        This is used in training to get a gradient, so it should return 
        a pytorch tensor which has been tracking the gradient.
        """
        pass

    @abstractmethod
    def get_expected_step_payoffs(self, p1, p2, prior, **kwargs):
        """
        Return the expected value of one step of the game.
        """
        pass

    def generate_test_priors(self):
        """
        Return priors against which to test, which may be 
        different from the player's priors.
        Envs can add more to this list.
        """
        priors = [] # [(name, prior)]
        priors.append(("Player1", self.prior_1))
        priors.append(("Player2", self.prior_2))
        unif_mix = [[(self.prior_1[a1][a2] + self.prior_2[a1][a2])/2 for a2 in [0,1]] for a1 in [0,1]]
        priors.append(("Unif_mix", unif_mix))
        return priors

    def select_prior(self, prior_n):
        """
        Some pre-set prior distributions to choose from.
        Environments don't actually need to use these.
        """
        self.preset_dists = [
            [[1/self.n_games for _ in range(self.n_types)] for _ in range(self.n_types)],
            # The following are for games with two types
            [[1., 0.], [0., 0.]],
            [[0., 1.], [0., 0.]],
            [[0., 0.], [1., 0.]],
            [[0., 0.], [0., 1.]],
            [[0.1, 0.4], [0.2, 0.3]],
            [[0.02, 0.94], [0.02, 0.02]],
            [[0.01, 0.97], [0.01, 0.01]],
        ]
        return self.preset_dists[prior_n]

    def sample_types(self, prior=None):
        prior = self.prior_1 if prior is None else prior
        tot, t1, r, done = 0, -1, random.random(), False
        while not done:
            t1 += 1
            games = prior[t1]
            t2 = -1
            for g_prob in games:
                t2 += 1
                tot += g_prob
                if tot >= r:
                    done = True
                    break
        return t1, t2

    def game_to_string(self, pfs):
        action_names = ["A", "B"]
        max_name = max([len(n) for n in action_names])
        top = "\n" + " " * max_name + "|"
        for n in action_names:
            top += n + "|"
        bar = "\n" + "-" * (len(top) - 1)
        mid = ""
        for i, n1 in enumerate(action_names):
            mid += "\n" + n1 + " " * (max_name - len(n1)) + "|"
            for j, n2 in enumerate(action_names):
                scores = str(pfs[i][j][0]) + ", " + str(pfs[i][j][1])
                mid += scores + " " * (len(n2) - len(scores)) + "|"
        return top + bar + mid

    def __str__(self):
        strs = []
        for i, gs in enumerate(self.pfs):
            for j, pfs in enumerate(gs):
                strs.append("\n{},{}\n".format(i, j) + self.game_to_string(pfs))
        return "\n".join(strs)

    def normal_form(self, prior=None):
        prior = prior if prior is not None else self.prior_1
        actions = [0, 1]
        rep_actions = [actions] * (2 * self.n_types)
        # a12 is action of player 1 for type 2
        rs = []
        for strats in itertools.product(*rep_actions):
            r1, r2 = 0, 0
            for t1, t2 in itertools.product(range(self.n_types), range(self.n_types)):
                r1 += prior[t1][t2] * self.pfs[t1][t2][strats[t1]][strats[self.n_types + t2]][0]
                r2 += prior[t1][t2] * self.pfs[t1][t2][strats[t1]][strats[self.n_types + t2]][1]
            rs.append((r1, r2))
        
        rowlen = len(actions)**self.n_types
        payoffs = [[rs[(p1 * rowlen)+p2] for p2 in range(rowlen)] for p1 in range(rowlen)]
        return payoffs

    def print_normalform(self, prior=None):
        print("\n".join(map(str, self.normal_form(prior=prior))))
    
    def get_pure_outcomes_as_points(self, prior=None):
        nform = self.normal_form(prior=prior)
        points = [rs for strat1 in nform for rs in strat1]
        return points

    def outcomes_polygon(self, prior=None):
        try:
            points = self.get_pure_outcomes_as_points(prior=prior)
            hull = ConvexHull(points)
            polygon_xs = [points[p][0] for p in hull.vertices]
            polygon_ys = [points[p][1] for p in hull.vertices]
            return polygon_xs, polygon_ys
        except scipy.spatial.qhull.QhullError:
            return [[],[]]
