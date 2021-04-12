from .communication_game import DistInfComms
from .mem1_game import DistInf, IncompleteFour

def get_game(name, **kwargs):
    if name == "DistInf":
        return DistInf(p=kwargs['p'], a=kwargs['a'], oneshot=kwargs['oneshot'])
    if name == "IncompFour":
        return IncompleteFour(prior_n=kwargs['prior_n'], oneshot=kwargs['oneshot'])
    if name == "DistInfComms":
        return DistInfComms(p=kwargs['p'], a=kwargs['a'])
    
    raise NotImplementedError("Game not implemented")