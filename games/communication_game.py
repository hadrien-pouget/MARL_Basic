import os
import random
from itertools import product
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from games.base_games import IncompleteInfoGame
from quicksave import QuickSaver

class BinCommunicationGame(IncompleteInfoGame):
    """
    This is a oneshot game, designed for two players, two types, and two actions.
    Players get types, then p1 sends a message (A or B), to p2, 
    then p2 to p1, then they simultaneously take an action (A or B)
    """
    def __init__(self, payoffs, prior_1=None, prior_2=None, prior_1_param=0, prior_2_param=0):
        super().__init__(payoffs, prior_1=prior_1, prior_2=prior_2, prior_1_param=prior_1_param, prior_2_param=prior_2_param)
        # 0: inital 1: after first message 2: after second message 3: done
        self.state = 0

    def gen_rand_policies(self):
        return torch.randn(10, requires_grad=True), torch.randn(12, requires_grad=True)

    def get_value(self, p1, p2, prior_1=None, prior_2=None, **kwargs):
        if prior_1 is None:
            prior_1 = self.prior_1

        if prior_2 is None:
            prior_2 = self.prior_2

        v1, _ = get_value_incomplete_bincomms_oneshot(self.pfs, p1, p2, prior_1)
        _, v2 = get_value_incomplete_bincomms_oneshot(self.pfs, p1, p2, prior_2)

        return v1, v2
    
    def get_expected_step_payoffs(self, p1, p2, prior, **kwargs):
        with torch.no_grad():
            return self.get_value(p1, p2, prior_1=prior, prior_2=prior, **kwargs)

    def calc_pt2s2(self, p2, t2, s2, pt1s1):
        S = 0 # p(s2_t2)
        for t1 in [0,1]:
            for s1 in [0,1]:
                ps2_t2s1 = p2[(t2*2)+s1] if s2==0 else 1 - p2[(t2*2)+s1]
                S += ps2_t2s1 * pt1s1[(t1*2)+s1]
        return S * 0.5

    def MI_ts(self, p1, p2):
        """
        Mutual information between type and signal for p1 and p2
        In this case, 0.6 is max; completely predictive

        Makes assumption that the probability of each type is 50%,
        which may not be generally true
        """
        ts = list(product([0,1], repeat=2))
        # p(t,s) = p(t)p(s|t)
        pt1s1 = [0.5 * (p1[t] if s==0 else 1-p1[t]) for t, s in ts]

        # p(s) = sum_t p(s, t)
        ps1 = [pt1s1[s]+pt1s1[s+2] for s in [0,1]]

        # p(t,s) = p(t)p(s|t)
        pt2s2 = [self.calc_pt2s2(p2, t2, s2, pt1s1) for t2, s2 in ts]

        # p(s) = sum_t p(s, t)
        ps2 = [pt2s2[s]+pt2s2[s+2] for s in [0,1]]

        MI1 = sum([pt1s1[(t*2)+s] * np.log(pt1s1[(t*2)+s]/(ps1[s]*0.5)) for t, s in ts])
        MI2 = sum([pt2s2[(t*2)+s] * np.log(pt2s2[(t*2)+s]/(ps2[s]*0.5)) for t, s in ts])

        return MI1, MI2

    def MI_as(self, p1, p2, prior=None):
        """
        MI between actions and messages
        """
        prior = self.prior_1 if prior is None else prior
        sstt = list(product([0,1], repeat=4))
        ssa = list(product([0,1], repeat=3))
        ss = list(product([0,1], repeat=2))
        tt = list(product([0,1], repeat=2))
        sa = list(product([0,1], repeat=2))

        ps1s2_t1t2 = [(p1[t1] if s1==0 else 1-p1[t1])*(p2[(t2*2)+s1] if s2==0 else 1 - p2[(t2*2)+s1]) for s1, s2, t1, t2 in sstt]

        ps1s2 = [sum([prior[t1][t2] * ps1s2_t1t2[s1*8+s2*4+t1*2+t2] for t1, t2 in tt]) for s1, s2 in ss]

        pa1_s1s2 = [0.5 * (p1[2+s1*2+s2] if a1==0 else 1 - p1[2+s1*2+s2]) + 0.5 * (p1[6+s1*2+s2] if a1==0 else 1 - p1[6+s1*2+s2]) for s1, s2, a1 in ssa]
        pa1s1s2 = [pa1_s1s2[s1*4+s2*2+a1]*ps1s2[s1*2+s2] for s1, s2, a1 in ssa]
        pa1 = [sum([pa1s1s2[s1*4+s2*2+a1] for s1, s2 in ss]) for a1 in [0,1]]
        # Between both messages and p1's action
        MI1 = sum([pa1s1s2[s1*4+s2*2+a1]*np.log(pa1s1s2[s1*4+s2*2+a1]/(pa1[a1]*ps1s2[s1*2+s2])) for s1, s2, a1 in ssa])


        pa2_s1s2 = [0.5 * (p2[4+s1*2+s2] if a2==0 else 1 - p2[2+s1*2+s2]) + 0.5 * (p2[8+s1*2+s2] if a2==0 else 1 - p2[6+s1*2+s2]) for s1, s2, a2 in ssa]
        pa2s1s2 = [pa2_s1s2[s1*4+s2*2+a2]*ps1s2[s1*2+s2] for s1, s2, a2 in ssa]
        pa2 = [sum([pa2s1s2[s1*4+s2*2+a2] for s1, s2 in ss]) for a2 in [0,1]]
        # Between both messages and p2's action
        MI2 = sum([pa2s1s2[s1*4+s2*2+a2]*np.log(pa2s1s2[s1*4+s2*2+a2]/(pa2[a2]*ps1s2[s1*2+s2])) for s1, s2, a2 in ssa])

        pa1s1 = [sum([pa1s1s2[s1*4+s2*2+a1] for s2 in [0,1]]) for s1, a1 in sa]
        pa1s2 = [sum([pa1s1s2[s1*4+s2*2+a1] for s1 in [0,1]]) for s2, a1 in sa]
        pa2s1 = [sum([pa2s1s2[s1*4+s2*2+a2] for s2 in [0,1]]) for s1, a2 in sa]
        pa2s2 = [sum([pa2s1s2[s1*4+s2*2+a2] for s1 in [0,1]]) for s2, a2 in sa]

        ps1 = [sum([ps1s2[s1*2+s2] for s2 in [0,1]]) for s1 in [0,1]]
        ps2 = [sum([ps1s2[s1*2+s2] for s1 in [0,1]]) for s2 in [0,1]]

        MIa1s1 = sum([pa1s1[s1*2+a1]*np.log(pa1s1[s1*2+a1]/(pa1[a1]*ps1[s1])) for s1, a1 in sa])
        MIa1s2 = sum([pa1s2[s2*2+a1]*np.log(pa1s2[s2*2+a1]/(pa1[a1]*ps2[s2])) for s2, a1 in sa])
        MIa2s1 = sum([pa2s1[s1*2+a2]*np.log(pa2s1[s1*2+a2]/(pa2[a2]*ps1[s1])) for s1, a2 in sa])
        MIa2s2 = sum([pa2s2[s2*2+a2]*np.log(pa2s2[s2*2+a2]/(pa2[a2]*ps2[s2])) for s2, a2 in sa])

        return MI1, MI2, MIa1s1, MIa1s2, MIa2s1, MIa2s2

    def get_equilibria(self, p1s, p2s, prior=None):
        """
        Given a list of p1s and p2s, return the equilibria

        Returned as a list [babble, coord, leader, comm, other]
        where each item in the list is a set of indices corresponding to
        the policies in that equilibrium
        """
        prior = self.prior_1 if prior is None else prior

        ### Mutual information between signals and types, and signals and actions
        MIts = [self.MI_ts(p1, p2) for p1, p2 in zip(p1s, p2s)]
        MIas = [self.MI_as(p1, p2) for p1, p2 in zip(p1s, p2s)]

        all_ps = set(range(len(p1s)))
        no_info1 = set([i for i in range(len(MIts)) if MIts[i][0] < 0.1])
        no_info2 = set([i for i in range(len(MIts)) if MIts[i][1] < 0.1])
        info1 = set([i for i in range(len(MIts)) if MIts[i][0] > 0.1])
        info2 = set([i for i in range(len(MIts)) if MIts[i][1] > 0.1])

        depends_a1s1 = set([i for i in range(len(MIas)) if MIas[i][2] > 0.1])
        depends_a1s2 = set([i for i in range(len(MIas)) if MIas[i][3] > 0.1])
        depends_a2s1 = set([i for i in range(len(MIas)) if MIas[i][4] > 0.1])
        depends_a2s2 = set([i for i in range(len(MIas)) if MIas[i][5] > 0.1])

        depends_boths_a1 = depends_a1s1.intersection(depends_a1s2)
        depends_boths_a2 = depends_a2s1.intersection(depends_a2s2)

        botha_depend_boths = depends_boths_a1.intersection(depends_boths_a2)

        ig_sig1  = set([i for i in range(len(MIas)) if MIas[i][0] < 0.1])
        ig_sig2  = set([i for i in range(len(MIas)) if MIas[i][1] < 0.1])

        no_info = no_info1.intersection(no_info2)
        info = info1.intersection(info2)
        ig_sig  = ig_sig1.intersection(ig_sig2)

        babble = no_info.intersection(ig_sig)
        coord = no_info.intersection(botha_depend_boths)
        leader = no_info1.intersection(depends_a1s2.intersection(depends_a2s2))
        comm = info

        res = [babble, coord, leader, comm]
        # Sanity check
        for r1 in range(len(res)):
            for r2 in range(r1+1, len(res)):
                if len(res[r1].intersection(res[r2])) != 0:
                    print("Overlap in sets of equilibria, {} {}".format(r1, r2))

        other = all_ps.difference(babble.union(coord).union(leader).union(comm))
        res.append(other)

        return res

    def plot_with_equilibria(self, folder):
        dic = QuickSaver().load_json_path(os.path.join('quick_saves', folder, 'Pols_res_0.json'))
        r1s = [x[0] for k, x in dic.items()]
        r2s = [x[1] for k, x in dic.items()]
        p1s = [x[2] for k, x in dic.items()]
        p2s = [x[3] for k, x in dic.items()]

        equi = self.get_equilibria(p1s, p2s)
        names = ["Babble", "Coord", "Leader", "Comm", "Other"]

        hues = []
        for i in range(len(r1s)):
            for e, eq in enumerate(equi):
                if i in eq:
                    hues.append(names[e])
                    break

        colours = {names[n]:'C{}'.format(n) for n in range(len(names))}
        sns.scatterplot(x=r1s, y=r2s, hue=hues, linewidth=1.3, palette=colours)

        polygon = self.outcomes_polygon()
        plt.fill(polygon[0], polygon[1], alpha=0.1, color='purple')

class SimpleDistInfComms(BinCommunicationGame):
    def __init__(self, a, prior_1_param, prior_2_param):
            """
            Implementing simplified version of game from 
            https://www.cambridge.org/core/journals/international-organization/article/abs/modeling-the-forms-of-international-cooperation-distribution-versus-information/F39FD45847A5593140E975E4EA2D580E
            with communication.
            p is the probability that they both prefer the same the outcome,
            a is the magnitude of preference for each outcome
            both players know p and a

            This is simplified in that the player's signals are the same iff the have the same preference
            (and different iff they're playing BoS)
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

            games = [[prefA, bos], [bos, prefB]]
            priors = []
            for p in [float(prior_1_param[0]), float(prior_2_param[0])]:
                priors.append(self.gen_prior(p))

            super().__init__(games, prior_1=priors[0], prior_2=priors[1])
            self.name = "SimpleDistInfComms"

    def gen_prior(self, p):
        return [[0.5*p, 0.5*(1-p)], [0.5*(1-p), 0.5*p]]

    def generate_test_priors(self):
        priors = super().generate_test_priors()
        for p in range(11):
            priors.append(("p_{}".format(p), self.gen_prior(p/10)))
        return priors

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
