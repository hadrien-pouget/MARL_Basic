from experiments import experiment
from games import get_game
from utils import seed
import numpy as np

GAME = "SimpleDistInfComms"

for prior_1 in range(0, 11, 1):
    prior_1_param = [prior_1/10]
    for prior_2 in range(0, 11, 1):
        prior_2_param = [prior_2/10]
        for a in range(1, 11, 1):
            config = {
                "a":a,
                "game":GAME,
                "gamma":0.96,
                "learning_rate":1,
                "oneshot":True,
                "prior_1_param":prior_1_param,
                "prior_2_param":prior_2_param,
                "seed":1234,
                "step_type":"naive",
                "test_ep":100,
                "train_ep":100,
                "training_rounds":40
            }

            seed(1234) # pretty sure this isn't like running each experiment alone with this seed
            env = get_game(GAME, prior_1_param=prior_1_param, prior_2_param=prior_2_param, a=a)
            save_folder = "{}_p{}{}_a{}_{}".format(GAME, prior_1, prior_2, a, 'naive')
            experiment(env, 'naive', 40, 0.96, 1.0, 100, True, 100, save_folder, config)
