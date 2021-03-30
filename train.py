import torch

def train_policies(env, training_rounds, value_func, step_func, train_ep, gamma, lr):
    p1s, p2s = [], []
    for n in range(training_rounds):
        print("Training round:", n, end='\r')
        p1, p2 = env.gen_rand_policy(), env.gen_rand_policy()
        p1, p2 = train(env, p1, p2, value_func, step_func, train_ep, gamma, lr)
        p1, p2 = torch.sigmoid(p1), torch.sigmoid(p2)
        p1s.append(p1)  
        p2s.append(p2)
    print()
    return p1s, p2s

def train(env, p1, p2, value_func, step_func, n_episodes, gamma, lr, verbose=0):
    """ 
    For a one-shot incomplete information game. p1 and p2 are a list 
    of numbers representing the pre-sigmoid probability for action 0.
    """
    if verbose > 1:
        print()
        print("---- Beginning training ----")
    for e in range(n_episodes):
        v1, v2 = value_func(env.pfs, p1, p2, env.dist, gamma)
        if verbose > 1:
            with torch.no_grad():
                print("E {:3} | v1: {:0.2f} | v2: {:0.2f} | p1: {:0.2f} {:0.2f} | p2: {:0.2f} {:0.2f}".format(e+1, v1.item(), v2.item(), *torch.sigmoid(p1).tolist(), *torch.sigmoid(p2).tolist()))
                # print("E {:2} | p1: {:0.2f} {:0.2f} | p2: {:0.2f} {:0.2f}".format(e+1, *p1.tolist(), *p2.tolist()))

        p1, p2 = step_func(p1, p2, v1, v2, lr)

    if verbose > 0:
        with torch.no_grad():
            v1, v2 = value_func(env.pfs, env.dist, p1, p2)
            print("E   F | v1: {:0.2f} | v2: {:0.2f} | p1: {:0.2f} {:0.2f} | p2: {:0.2f} {:0.2f}".format(v1.item(), v2.item(), *torch.sigmoid(p1).tolist(), *torch.sigmoid(p2).tolist()))
    return p1.detach().clone(), p2.detach().clone()
