from .communication_game import DistInfComms
from .mem1_game import DistInf, IncompleteFour

ALL_GAMES = ["DistInf", "IncompFour", "DistInfComms"]
def get_game(game, **kwargs):
    if game == "DistInf":
        return DistInf(p=kwargs['p'], a=kwargs['a'], oneshot=kwargs['oneshot'])
    if game == "IncompFour":
        return IncompleteFour(prior_n=kwargs['prior_n'], oneshot=kwargs['oneshot'])
    if game == "DistInfComms":
        return DistInfComms(p=kwargs['p'], a=kwargs['a'])
    
    raise NotImplementedError("Game not implemented")
