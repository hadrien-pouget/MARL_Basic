from .communication_game import SimpleDistInfComms

ALL_GAMES = ["SimpleDistInfComms"]
def get_game(game, **kwargs):
    if game == "SimpleDistInfComms":
        return SimpleDistInfComms(prior_1_param=kwargs['prior_1_param'], prior_2_param=kwargs['prior_2_param'], a=kwargs['a'])
    
    raise NotImplementedError("Game not implemented")
