import torch

def get_value_incomplete_oneshot(payoffs, dist, p1, p2):
    # Turn into probabilities
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)

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

def get_value_iterated(payoffs, p1, p2, gamma):
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
    r1s = torch.tensor([r1 for a1 in payoffs for r1, r2 in a1], dtype=torch.float64)
    r2s = torch.tensor([r2 for a1 in payoffs for r1, r2 in a1], dtype=torch.float64)
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