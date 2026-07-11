# Technology 1 — Neuroevolution and Technological Development

This experiment models technological development as an evolutionary strategy.

A population of agents lives on a continuous, uneven terrain. Each agent has a small neural network that decides whether to move in one of eight directions or spend energy modifying the terrain. Agents reproduce according to two variables: the maximum height they reach and the energy they preserve. Their neural networks are inherited, mutated, and sometimes crossed with those of other successful agents.

## Main idea

The terrain represents an economic or technological environment. Height represents access to a more productive or advantageous position. Energy represents limited resources. Agents may either:

- exploit the existing terrain by climbing naturally;
- conserve energy for reproduction;
- invest energy in modifying the terrain;
- benefit from infrastructure created by previous generations.

Terrain modification follows a support rule. To raise the center of a 3×3 area by one level, the eight surrounding cells must already support that level. Repeated construction therefore produces terraces or pyramid-like structures rather than isolated vertical towers.

This creates a trade-off between short-term efficiency and long-term technological investment. Building is more expensive than moving, but it may create paths that later generations can exploit.

## Neuroevolution

Each agent observes:

- the relative heights of the surrounding 3×3 area;
- its remaining energy;
- its current normalized height;
- whether construction is currently possible;
- the relative cost of construction;
- the progress of the current generation.

The neural network outputs one of nine actions: eight movement directions or one construction action.

The most successful neural strategies produce more descendants. Over many generations, the population may evolve different balances between exploration, climbing, conservation, and technological investment.

## Installation

```bash
pip install numpy scipy matplotlib
```

## Run

```bash
python neuroevolution_technology.py
```

Example with custom parameters:

```bash
python neuroevolution_technology.py \
  --generations 200 \
  --population 350 \
  --turns 70 \
  --terrain-size 150 \
  --seed 12
```

## Outputs

The program displays:

1. the final effective terrain;
2. the accumulated constructed layer;
3. the evolution of maximum and average height;
4. the relationship between remaining energy and technological investment.

## Interpretation

The simulation is not intended as a realistic economic forecast. It is a conceptual model for studying how technology may emerge when agents face scarcity, inherited infrastructure, uncertain environments, and competition for reproduction.
