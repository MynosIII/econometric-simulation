# Pilot 1: Evolutionary Hill Climber

This pilot simulates a population of simple agents trying to reach higher areas on a continuous, irregular terrain.

Each agent has a small genome that controls four behavioral traits:

- preference for moving uphill;
- tendency to continue in the same direction;
- level of random exploration;
- preference for steep or gentle slopes.

At the end of each generation, agents that finish at higher positions have a greater probability of reproducing. Their descendants inherit similar genomes with small random mutations. Over many generations, the population can develop better climbing strategies.

This is an evolutionary algorithm rather than supervised machine learning. Agents do not learn during their lifetime; adaptation happens across generations through selection and mutation.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Matplotlib

Install the dependencies with:

```bash
pip install numpy scipy matplotlib
```

## Run

From the repository root:

```bash
python "piloto 1/evolutionary_hill_climber.py"
```

The program prints the best and average height every ten generations. When the simulation finishes, it displays:

1. the terrain and the path followed by the best agent;
2. the best and average height reached across generations;
3. the genome of the best agent found.

## Main parameters

The parameters can be changed in the `__main__` section of the script:

- `generations`: number of evolutionary cycles;
- `population_size`: number of agents per generation;
- `moves_per_generation`: number of movements available to each agent;
- `mutation_rate`: size of the random genetic changes;
- `seed`: random seed used to reproduce the same experiment.

A useful next step would be replacing the four-parameter genome with a small neural network. The network could receive nearby terrain heights as inputs and choose one of eight possible movement directions.