from experiments import experiment
from games import get_game
from utils import seed
import numpy as np

for p in range(0, 11, 1):
    p = p/10
    for a in range(1, 11, 1):
        for step_type in ['lola', 'naive']:
            config = {
                "a":a,
                "dist_n":0,
                "game":"DistInf",
                "gamma":0.96,
                "learning_rate":1,
                "oneshot":True,
                "p":p,
                "seed":1234,
                "step_type":step_type,
                "test_ep":100,
                "train_ep":100,
                "training_rounds":40
            }
            seed(1234) # pretty sure this isn't like running each experiment alone with this seed
            env = get_game("DistInf", oneshot=True, p=p, a=a)
            save_folder = "DistInf_p{}_a{}_{}".format(p, a, step_type)
            experiment(env, step_type, 40, 0.96, 1.0, 100, True, 100, save_folder, config)
