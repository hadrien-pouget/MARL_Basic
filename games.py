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

def get_game(game):
    if game == 'PD':
        return PD
    elif game == 'BoS':
        return BoS
