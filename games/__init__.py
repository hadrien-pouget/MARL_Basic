from .communication_game import SimpleDistInfComms, DistInfComms
from .mem1_games import DistInf, IncompleteFour

ALL_GAMES = ["IncompFour", "SimpleDistInfComms", "DistInf", "DistInfComms"]
def get_game(game, **kwargs):
    if game == "DistInf":
        return DistInf(p=kwargs['p'], a=kwargs['a'], oneshot=kwargs['oneshot'])
    if game == "DistInfComms":
        return DistInfComms(p=kwargs['p'], a=kwargs['a'])
    if game == "IncompFour":
        return IncompleteFour(prior_1_param=kwargs['prior_1_param'], prior_2_param=kwargs['prior_2_param'], oneshot=kwargs['oneshot'])
    if game == "SimpleDistInfComms":
        return SimpleDistInfComms(prior_1_param=kwargs['prior_1_param'], prior_2_param=kwargs['prior_2_param'], a=kwargs['a'])
    
    raise NotImplementedError("Game not implemented")
