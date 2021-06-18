This code allows you to define incomplete information games, train agents to play them, and look at the results.

The state of the code is a bit messy, and in many places assumes that these are two player games.

# Games
The games should follow the interface set out in 
```
games/base_games.py. 
```
As of now, the only type of game implemented is incomplete information games (bayesian games), where the players may have different priors (over the same games). The games may be oneshot or repeating.

The games can follow pretty complex structures beyond this (such as communication).

# Training
Can use different training steps, based on the gradients from calculating the value:
- naive is a simple step in the direction of the gradient
- LOLA is a step done as in [Learning with Opponent-Learning Awareness](https://arxiv.org/pdf/1709.04326.pdf), where the step considers the opponent's update as well.

# Testing
test.py contains code for testing the policies (although some of the testing code is implemented on the game's end.)

# Utils
A lot of generally useful functions can be found in utils.py. Similarly, some functions useful for plotting results 
are contained in plot_utils.py.

# Results
disting_pa_experiments.py shows how to run some experiments, in the game described [here](https://www.jstor.org/stable/2706964). The agents are playing BoS with probability 1-*p*, or both prefer the same action with probability *p*. If they both prefer the same action, they may either both prefer action A (with probability *p*/2) or B (with probability *p*/2). While in all these games, coordinating is better than failing to coordinate, the intesity with which the agents prefer an action (either opposite actions in BoS, or the same if they both prefer action A or B) is determined by some value *a*. Each player gets a one-bit signal containing some information about the game they are playing. They each in turn send a one-bit signal to the other, before simultaneously selecting actions.

Google colabs of these results are available here:
- [here](https://colab.research.google.com/drive/1CfHrlJfYntGWMOyJhbRX6juJ79x-gZMc?usp=sharing) for payoff profiles of many agents training to play the game, with both normal play (play against the opponent they trained with) and cross-play (play against a different opponent) 
- [here](https://colab.research.google.com/drive/15fQ69Ov8mX4-KAF7Mck5_PUXpz-7hNmf?usp=sharing) on identifying the different equilibria that arise through training
- [here](https://colab.research.google.com/drive/1VnwucmJsPToFWVkzIhzOHfB4np5JjeVB?usp=sharing) to see the welfare achieved by different equilibria 
- [here](https://colab.research.google.com/drive/1D9PPSYJQwR8sfbeiF8ShW3ty0oxIr58Z?usp=sharing) to see what training agents with differing priors, and evaluating with respect to a third prior looks like
