from experiments import experiment
from games import get_game
from utils import seed
import numpy as np

def simpledistinfcomms():
    """
    SimpleDistInfComms environment with different priors for each player
    """
    GAME = "SimpleDistInfComms"
    STEP_TYPES = ['naive']

    for prior_1 in range(0, 11, 1):
        prior_1_param = [prior_1/10]
        for prior_2 in range(0, 11, 1):
            prior_2_param = [prior_2/10]
            for a in [1,2,5,10]:
                for st in STEP_TYPES:
                    config = {
                        "a":a,
                        "game":GAME,
                        "gamma":0.96,
                        "learning_rate":1,
                        "oneshot":True,
                        "prior_1_param":prior_1_param,
                        "prior_2_param":prior_2_param,
                        "seed":1234,
                        "step_type":st,
                        "test_ep":100,
                        "train_ep":100,
                        "training_rounds":40
                    }

                    seed(1234) # pretty sure this isn't like running each experiment alone with this seed
                    env = get_game(GAME, prior_1_param=prior_1_param, prior_2_param=prior_2_param, a=a)
                    save_folder = "{}_p{}_{}_a{}_{}".format(GAME, prior_1, prior_2, a, st)
                    experiment(env, st, 40, 0.96, 1.0, 100, True, 100, save_folder, 'cuda', config)

def distinfcomms():
    """
    DistInfComms experiment with the same prior for both players
    """
    GAME = "DistInfComms"
    STEP_TYPES = ['naive', 'lola']

    for p in range(0, 11, 1):
        p = p/10
        for a in [1,2,5,10]:
            for st in STEP_TYPES:
                config = {
                    "a":a,
                    "game":GAME,
                    "gamma":0.96,
                    "learning_rate":1,
                    "oneshot":True,
                    "p":p,
                    "seed":1234,
                    "step_type":st,
                    "test_ep":100,
                    "train_ep":100,
                    "training_rounds":40
                }

                seed(1234) # pretty sure this isn't like running each experiment alone with this seed
                env = get_game(GAME, p=p, a=a)
                save_folder = "{}_p{}_a{}_{}".format(GAME, p, a, st)
                experiment(env, st, 40, 0.96, 1.0, 100, True, 100, save_folder, 'cuda', config)

if __name__ == '__main__':
    distinfcomms()
