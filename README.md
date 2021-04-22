# Battleship AI

## Installation
1. Navigate to the directory containing the code
2. Use `pipenv` to shell into a virtual environment: `pipenv shell` (if you do not have `pipenv`, run `pip install pipenv` first)
3. Install all the dependencies with `pipenv install`

## Evaluating agents' performance
run `python bs_gym_agents.py` to evaluate the performance of the Random Agent, Random w/ Preferred Actions Agent, Probabilistic Agent, and Particle Filter Agent. This command has each agent play 100 games (it may take around 10 minutes for the Particle Filter Agent to finish). The output will look like this:

```
Agents to be evaluated: ['Random Agent', 'Random w/ Preferred Actions Agent', 'Probabilistic Agent', 'Particle Filter Agent']
Evaluating Random Agent agent: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 181.72it/s]
Evaluating Random w/ Preferred Actions Agent agent: 100%|████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 270.29it/s] 
Evaluating Probabilistic Agent agent: 100%|███████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:23<00:00,  4.29it/s] 
Evaluating Particle Filter Agent agent: 100%|█████████████████████████████████████████████████████████████████████████████████████| 100/100 [09:47<00:00,  5.87s/it]
                                    min  median    max   mean        std  avg_time  episodes
Random Agent                       79.0    97.5  100.0  95.64   4.604609  0.005543     100.0
Random w/ Preferred Actions Agent  25.0    60.0   94.0  60.23  15.356935  0.003700     100.0
Probabilistic Agent                27.0    46.5   66.0  46.45   9.263129  0.233110     100.0
Particle Filter Agent              29.0    46.0   70.0  47.71   8.680985  5.871444     100.0
```

where the most relevant indicators are `mean` (the average number of shots taken per game, with the least possible being 17 to sink all 5 ships) and `avg_time` (the average time taken to complete a game).

## Optional Jupyter notebook
The notebook `figure_generation.ipynb` was used to generate the figures we included in the paper. Running it in Google Colab was necessary to make use of their robust GPUs, in evaluating the deep Q agents. Each cell has a comment explaining what it does.  Running certain parts of the notebook requires access to trained deep Q models which are not included here, but you may train new models by running the corresponding cells in the notebook. In order to generate the figures, either edit the code to not use the deep Q agents, or run the provided code to train new models.