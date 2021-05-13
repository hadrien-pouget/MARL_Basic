Define incomplete information games, train agents to play them, and look at the results.

The state of the code is a bit messy, and in many places assumes that these are two player games.

# Games
The games should follow the interface set out in 
```
games/base_games.py. 
```
As of now, the only type of game implemented is incomplete information games (bayesian games), where the players may have different priors (over the same games). The games may be oneshot or repeating.

The games can follow pretty complex structures beyond this (such as communication).

# Training
Can use different training steps, based on the gradients from calculating the value. So far, naive and lola gradient steps are implemented.

# Testing
test.py contains code for testing the policies (although some of the testing code is implemented on the game's end.)

# Visualising
Can be unique to each env, using jupyter notebooks for now.

# Utils
A lot of generally useful functions for visualising, etc are put utils.py

# Notes
I've removed:
- some games
- empirical testing
- ability to treat game like an environment (with a step and reset function)

from the code, when making it use separate priors. These could pretty easily be added back in.
