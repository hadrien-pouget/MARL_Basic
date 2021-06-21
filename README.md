# Multi-Agent Reinforcement Learning with Games of Incomplete Information

This code was written to allow for experimentation in evaluating agents playing games of incomplete
information. It contains modules for defining games, training agents to play those games, testing 
agents, and saving/visualising results. It includes a generic command-line interface for running experiments
(`experiments.py`), and a script using this to run concrete experiments (`disting_pa_experiments.py`).

Currently, the code largely assumes that these are two player games, usually with two actions available
to each player. In some places, it remains quite messy. It also assumes understanding of bayesian games, and multi-agent reinforcement learning.

The code has only been tested on windows.

# Set-Up
This repo includes an `environment.yml` file, to 
recreate the development environment. This environment 
may include more packages than are actually required.

To create the environment using conda:
```
conda env create -f environment.yml
```

To get started using the code, you can take a look at [Easy Places to Start](#easy-places-to-start).

# Code Structure

The code separates the definition of the games (and game-specific functions), the training logic, the saving 
tools, the testing tools, and the visualisation tools.

| File | Function |
| ----------- | ----------- |
| `games/` | Framework and concrete examples for implementing games |
| `environment.yml` | Includes required python packages |
| `plot_utils` | Tools for plotting results, some of which are fairly specific |
| `quicksave.py` | A generic saving tool I use in my projects |
| `test.py` | Functions for getting expected payoffs of agents |
| `train.py` | Jointly train randomly initialised policies |
| `train_steps.py` | Possible gradient steps used during training |
| `utils.py` | Mostly tools for saving and loading |
| `welfare_functions.py` | Implementations of social welfare functions, useful for analysis |

## Games
Code relevant to the games is contained in the `games` folder. The games should follow the interface set out in `games/base_games.py`. `mem1_games.py` contains implementation of repeated games, where agents are given the actions chosen in the previous step to make their next decision. Oneshot games are implemented as a special case of this. `communication_game.py` contains an implementation of oneshot games including communication channels.

## Training
Two different gradient steps are included, based on the gradients from calculating the expected payoff during training.
- "Naive" is a simple step in the direction of the gradient
- "LOLA" is a step done as in [Learning with Opponent-Learning Awareness](https://arxiv.org/pdf/1709.04326.pdf), where the step considers the opponent's update as well

## Testing
`test.py` contains code for testing the policies. Most of the 
code for exact calculations is done on the game's end, but this 
file also includes some code for empirical evaluation.

## Easy Places to Start
`experiments.py` includes a command-line interface for running experiments, 
which includes training agents, testing them, and saving the results/trained 
agents. `distinf_pa_experiments.py` includes which uses this interface to 
run larger experiments. Some Google Colabs are available in the 
[Results](#results) section, which show how this code can be used 
to get in-depth visualisations of the results.

When using `experiments.py`, results will be saved in an automatically generated folder called `quick_saves`. Each time an experiments is run, it will be given its own subfolder, including a `config.json` file to track the parameters used for that experiment.

# Results
`disting_pa_experiments.py` shows how to run some experiments, using the game described in [Modeling the Forms of International Cooperation: Distribution Versus Information](https://www.jstor.org/stable/2706964). 

## The Game
The agents are playing BoS with probability 1-*p*, or both prefer the same action with probability *p*. If they both prefer the same action, they may either both prefer action A (with probability *p*/2) or B (with probability *p*/2). 

While in all these games coordinating is better than failing to coordinate, the intensity with which the agents prefer an action (either opposite actions in BoS, or the same if they both prefer action A or B) is determined by some value *a*. 

Each player gets a one-bit signal containing some information about the game they are playing, which allows them to separately update their beliefs. They each in turn send a one-bit signal to the other (to communicate), before simultaneously selecting actions.

## Visualisations of Results

Google colabs of these results are available here:
- [here](https://colab.research.google.com/drive/1CfHrlJfYntGWMOyJhbRX6juJ79x-gZMc?usp=sharing) for payoff profiles of many agents trained to play the game, with both normal play (play against the opponent they trained with) and cross-play (play against a different opponent) 
- [here](https://colab.research.google.com/drive/15fQ69Ov8mX4-KAF7Mck5_PUXpz-7hNmf?usp=sharing) on identifying the different equilibria that arise through training
- [here](https://colab.research.google.com/drive/1VnwucmJsPToFWVkzIhzOHfB4np5JjeVB?usp=sharing) to see the welfare achieved by different equilibria 
- [here](https://colab.research.google.com/drive/1D9PPSYJQwR8sfbeiF8ShW3ty0oxIr58Z?usp=sharing) to see what training agents with differing priors, and evaluating with respect to a third prior looks like
