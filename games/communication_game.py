import random

import torch

from games.base_games import IncompleteInfoGame


class BinCommunicationGame(IncompleteInfoGame):
    """
    This is a oneshot game, designed for two players, two types, and two actions.
    Players get types, then p1 sends a message (A or B), to p2, 
    then p2 to p1, then they simultaneously take an action (A or B)
    """
    def __init__(self, payoffs, prior=None, prior_n=0):
        super().__init__(payoffs, prior=prior, prior_n=0)
        # 0: inital 1: after first message 2: after second message 3: done
        self.state = 0

    def reset(self):
        self.state = 0
        self.types = self.sample_types()
        return self.types

    def step(self, a1, a2):
        """
        When both take actions, without communicating
        """
        return self.pfs[self.types[0]][self.types[1]][a1][a2]

    def gen_rand_policies(self):
        return torch.randn(10, requires_grad=True), torch.randn(12, requires_grad=True)

    def get_value(self, p1, p2, prior=None, **kwargs):
        if prior is None:
            prior = self.prior
        return get_value_incomplete_bincomms_oneshot(self.pfs, p1, p2, prior)

    def play_game(self, p1, p2):
        types = self.reset()

        # p1 picks signal
        s1 = 0 if random.random() < get_param_p1(p1, types[0]) else 1

        # p2 picks signal
        s2 = 0 if random.random() < get_param_p2(p2, types[1], s1) else 1

        # pick actions
        a1 = 0 if random.random() < get_param_p1(p1, types[0], s1, s2) else 1
        a2 = 0 if random.random() < get_param_p2(p2, types[1], s1, s2) else 1

        return self.step(a1, a2)

class DistInfComms(BinCommunicationGame):
    def __init__(self, p, a):
        """
        Implementing game from https://www.cambridge.org/core/journals/international-organization/article/abs/modeling-the-forms-of-international-cooperation-distribution-versus-information/F39FD45847A5593140E975E4EA2D580E
        with communication.
        p is the probability that they both prefer the same the outcome,
        a is the magnitude of preference for each outcome
        both players know p and a
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
        super().__init__(games, prior=prior)
        self.name = "DistInfComms"

def get_param_p1(p1, t, sent=None, rec=None):
    """
    sent and rec - should have both or neither be None
    """
    # History: type, sent, received
    # t1 t2 t1s1s1 t1s1s2 t1s2s1 t1s2s2 t2s1s1 ...
    if sent is None:
        return p1[t]
    
    ind = 2
    ind += (4*t) + (sent*2) + rec
    return p1[ind]

def get_param_p2(p2, t, rec, sent=None):
    # History: type, received, sent
    # t1s1 t1s2 t2s1 t2s2 t1s1s1 t1s1s2 t1s2s1 t1s2s2 t2s1s1 ...
    if sent is None:
        return p2[(t*2) + rec]
    
    ind = 4
    ind += (4*t) + (rec*2) + sent
    return p2[ind]

def game_value(payoffs, p1, p2):
    """
    p1 is probability of player 1 taking action A
    """
    p1_a12 = torch.stack([p1, p1, 1 - p1, 1 - p1])
    p2_a12 = torch.stack([p2, 1 - p2, p2, 1 - p2])
    p_outcomes = p1_a12 * p2_a12
    # size: [4,] probability of [AA, AB, BA, BB] from s0
    
    r1s = torch.tensor([r1 for a1 in payoffs for r1, r2 in a1], dtype=torch.float)
    r2s = torch.tensor([r2 for a1 in payoffs for r1, r2 in a1], dtype=torch.float)
    # payoffs for AA, AB, BA, BB

    v1 = torch.dot(p_outcomes, r1s)
    v2 = torch.dot(p_outcomes, r2s)

    return v1, v2

def get_value_comm1(payoffs, p1, p2, ts, s1):
    """
    Value given player 1's signal
    """
    vs_per_game_1 = []
    vs_per_game_2 = []

    # Iterate over s2's signal
    for s2 in [0, 1]:
        v1, v2 = game_value(payoffs, get_param_p1(p1, ts[0], s1, s2), get_param_p2(p2, ts[1], s1, s2))
        vs_per_game_1.append(v1)
        vs_per_game_2.append(v2)

    vs_per_game_1 = torch.stack(vs_per_game_1)
    vs_per_game_2 = torch.stack(vs_per_game_2)

    ps0 = get_param_p2(p2, ts[1], s1)
    pss = torch.stack([ps0, 1-ps0])
    # Probability of p1 giving signal 0 and signal 1

    v1 = torch.dot(vs_per_game_1, pss)
    v2 = torch.dot(vs_per_game_2, pss)

    return v1, v2

def get_value_bincomms_oneshot(payoffs, p1, p2, ts):
    """
    Value given type
    """
    vs_per_game_1 = []
    vs_per_game_2 = []

    # Iterate over p1's signals
    for s1 in [0, 1]:
        v1, v2 = get_value_comm1(payoffs, p1, p2, ts, s1)
        vs_per_game_1.append(v1)
        vs_per_game_2.append(v2)

    vs_per_game_1 = torch.stack(vs_per_game_1)
    vs_per_game_2 = torch.stack(vs_per_game_2)

    ps0 = get_param_p1(p1, ts[0])
    pss = torch.stack([ps0, 1-ps0])
    # Probability of p1 giving signal 0 and signal 1

    v1 = torch.dot(vs_per_game_1, pss)
    v2 = torch.dot(vs_per_game_2, pss)

    return v1, v2


def get_value_incomplete_bincomms_oneshot(payoffs, p1, p2, dist):
    # Turn into probabilities
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)

    vs_per_game_1 = []
    vs_per_game_2 = []

    # Iterate over type combinations
    for t1, t2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        v1, v2 = get_value_bincomms_oneshot(payoffs[t1][t2], p1, p2, (t1, t2))
        vs_per_game_1.append(v1)
        vs_per_game_2.append(v2)

    vs_per_game_1 = torch.stack(vs_per_game_1)
    vs_per_game_2 = torch.stack(vs_per_game_2)
    dist = torch.tensor(dist).reshape(-1)

    v1 = torch.dot(vs_per_game_1, dist)
    v2 = torch.dot(vs_per_game_2, dist)

    return v1, v2

