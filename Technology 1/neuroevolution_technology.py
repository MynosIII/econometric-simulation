import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class Config:
    terrain_size: int = 120
    terrain_seed: int = 42
    simulation_seed: int = 7
    population_size: int = 220
    generations: int = 120
    turns_per_generation: int = 55
    starting_energy: float = 100.0
    move_cost: float = 0.55
    uphill_cost_factor: float = 0.35
    failed_action_cost: float = 0.12
    improve_base_cost: float = 2.5
    improve_level_cost: float = 0.9
    improve_height_cost: float = 0.10
    build_increment: float = 1.0
    max_build_level: int = 30
    max_climb_step: float = 2.2
    hidden_size: int = 24
    exploration_temperature: float = 0.35
    elite_fraction: float = 0.05
    mutation_rate: float = 0.10
    mutation_scale: float = 0.12
    crossover_rate: float = 0.35
    height_weight: float = 2.0
    energy_weight: float = 1.4
    log_every: int = 10


def generate_base_terrain(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    large = gaussian_filter(rng.normal(size=(size, size)), sigma=18)
    medium = gaussian_filter(rng.normal(size=(size, size)), sigma=7)
    small = gaussian_filter(rng.normal(size=(size, size)), sigma=2)

    for layer in (large, medium, small):
        layer /= max(np.std(layer), 1e-9)

    terrain = 7.5 * large + 2.8 * medium + 0.45 * small
    y = np.linspace(0, 1, size)[:, None]
    x = np.linspace(0, 1, size)[None, :]
    variance_map = (
        0.55
        + 0.85 * y
        + 0.20 * np.exp(-((x - 0.72) ** 2 + (y - 0.28) ** 2) / 0.035)
    )

    terrain *= variance_map
    terrain -= terrain.min()
    terrain /= max(terrain.max(), 1e-9)
    return terrain * 20.0


class World:
    def __init__(self, base: np.ndarray, config: Config):
        self.base = base.astype(float)
        self.built = np.zeros_like(base, dtype=np.int16)
        self.config = config

    @property
    def height(self) -> np.ndarray:
        return self.base + self.built * self.config.build_increment

    def inside(self, row: int, col: int) -> bool:
        rows, cols = self.base.shape
        return 0 <= row < rows and 0 <= col < cols

    def can_improve(self, row: int, col: int) -> bool:
        rows, cols = self.base.shape
        if row <= 0 or row >= rows - 1 or col <= 0 or col >= cols - 1:
            return False

        current_level = int(self.built[row, col])
        if current_level >= self.config.max_build_level:
            return False

        neighborhood = self.built[row - 1:row + 2, col - 1:col + 2]
        support_mask = np.ones((3, 3), dtype=bool)
        support_mask[1, 1] = False
        return bool(np.all(neighborhood[support_mask] >= current_level))

    def improvement_cost(self, row: int, col: int) -> float:
        level = float(self.built[row, col])
        effective_height = float(self.height[row, col])
        return (
            self.config.improve_base_cost
            + self.config.improve_level_cost * level
            + self.config.improve_height_cost * effective_height
        )

    def improve(self, row: int, col: int) -> bool:
        if not self.can_improve(row, col):
            return False
        self.built[row, col] += 1
        return True


class NeuralGenome:
    def __init__(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    @classmethod
    def random(cls, rng: np.random.Generator, input_size: int, hidden_size: int, output_size: int):
        return cls(
            rng.normal(0, np.sqrt(2 / input_size), size=(hidden_size, input_size)),
            np.zeros(hidden_size),
            rng.normal(0, np.sqrt(2 / hidden_size), size=(output_size, hidden_size)),
            np.zeros(output_size),
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        hidden = np.tanh(self.w1 @ inputs + self.b1)
        return self.w2 @ hidden + self.b2

    def copy(self):
        return NeuralGenome(self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())

    def mutate(self, rng: np.random.Generator, rate: float, scale: float) -> None:
        for parameter in (self.w1, self.b1, self.w2, self.b2):
            mask = rng.random(parameter.shape) < rate
            parameter += mask * rng.normal(0, scale, size=parameter.shape)

    @staticmethod
    def crossover(a, b, rng: np.random.Generator):
        def blend(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            alpha = rng.random(x.shape)
            return alpha * x + (1 - alpha) * y

        return NeuralGenome(
            blend(a.w1, b.w1),
            blend(a.b1, b.b1),
            blend(a.w2, b.w2),
            blend(a.b2, b.b2),
        )


MOVES = np.array([
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],            [0, 1],
    [1, -1],  [1, 0],   [1, 1],
], dtype=int)


@dataclass
class Agent:
    genome: NeuralGenome
    position: np.ndarray
    energy: float
    initial_energy: float
    alive: bool = True
    max_height: float = -np.inf
    successful_improvements: int = 0
    successful_moves: int = 0
    path: List[Tuple[int, int]] = field(default_factory=list)

    def initialize(self, world: World) -> None:
        row, col = self.position
        self.max_height = float(world.height[row, col])
        self.path = [(int(row), int(col))]

    def observe(self, world: World, turn: int, total_turns: int) -> np.ndarray:
        row, col = self.position
        height = world.height
        rows, cols = height.shape
        center = float(height[row, col])
        patch = np.empty((3, 3), dtype=float)

        for i, dr in enumerate((-1, 0, 1)):
            for j, dc in enumerate((-1, 0, 1)):
                rr = int(np.clip(row + dr, 0, rows - 1))
                cc = int(np.clip(col + dc, 0, cols - 1))
                patch[i, j] = height[rr, cc] - center

        patch /= max(float(height.max() - height.min()), 1.0)

        return np.concatenate([
            patch.ravel(),
            np.array([
                np.clip(self.energy / max(self.initial_energy, 1e-9), 0.0, 1.5),
                (center - float(height.min())) / max(float(height.max() - height.min()), 1e-9),
                1.0 if world.can_improve(row, col) else 0.0,
                world.improvement_cost(row, col) / max(self.initial_energy, 1e-9),
                turn / max(total_turns - 1, 1),
            ]),
        ])

    def choose_action(self, world: World, turn: int, total_turns: int, rng: np.random.Generator, temperature: float) -> int:
        logits = self.genome.forward(self.observe(world, turn, total_turns))
        if temperature <= 0:
            return int(np.argmax(logits))

        scaled = logits / temperature
        scaled -= np.max(scaled)
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum()
        return int(rng.choice(len(logits), p=probabilities))

    def act(self, action: int, world: World) -> None:
        if not self.alive:
            return

        row, col = self.position

        if action < 8:
            dr, dc = MOVES[action]
            nr, nc = int(row + dr), int(col + dc)

            if not world.inside(nr, nc):
                self.energy -= world.config.failed_action_cost
            else:
                difference = float(world.height[nr, nc] - world.height[row, col])
                if difference > world.config.max_climb_step:
                    self.energy -= world.config.failed_action_cost
                else:
                    cost = world.config.move_cost + max(difference, 0.0) * world.config.uphill_cost_factor
                    if self.energy >= cost:
                        self.energy -= cost
                        self.position = np.array([nr, nc], dtype=int)
                        self.successful_moves += 1
                        self.path.append((nr, nc))
                    else:
                        self.energy = 0.0
        else:
            cost = world.improvement_cost(row, col)
            if self.energy >= cost and world.can_improve(row, col):
                self.energy -= cost
                world.improve(row, col)
                self.successful_improvements += 1
            else:
                self.energy -= world.config.failed_action_cost

        self.energy = max(self.energy, 0.0)
        row, col = self.position
        self.max_height = max(self.max_height, float(world.height[row, col]))
        self.alive = self.energy > 0


def random_position(world: World, rng: np.random.Generator) -> np.ndarray:
    rows, cols = world.base.shape
    return np.array([rng.integers(1, rows - 1), rng.integers(1, cols - 1)], dtype=int)


def create_population(world: World, config: Config, rng: np.random.Generator) -> List[Agent]:
    population = []
    for _ in range(config.population_size):
        agent = Agent(
            genome=NeuralGenome.random(rng, input_size=14, hidden_size=config.hidden_size, output_size=9),
            position=random_position(world, rng),
            energy=config.starting_energy,
            initial_energy=config.starting_energy,
        )
        agent.initialize(world)
        population.append(agent)
    return population


def fertility(population: List[Agent], world: World, config: Config) -> np.ndarray:
    heights = np.array([agent.max_height for agent in population])
    energies = np.array([agent.energy for agent in population])
    relative_height = (heights - world.height.min()) / max(float(world.height.max() - world.height.min()), 1e-9)
    relative_energy = np.clip(energies / max(config.starting_energy, 1e-9), 0.0, 1.0)
    score = (
        (0.05 + relative_height) ** config.height_weight
        * (0.05 + relative_energy) ** config.energy_weight
    )
    return score + 1e-12


def reproduce(population: List[Agent], world: World, config: Config, rng: np.random.Generator):
    scores = fertility(population, world, config)
    probabilities = scores / scores.sum()
    elite_count = max(1, round(config.population_size * config.elite_fraction))
    ranked = np.argsort(scores)[::-1]
    new_population = []

    for index in ranked[:elite_count]:
        child = Agent(
            genome=population[index].genome.copy(),
            position=random_position(world, rng),
            energy=config.starting_energy,
            initial_energy=config.starting_energy,
        )
        child.initialize(world)
        new_population.append(child)

    while len(new_population) < config.population_size:
        parent_a = population[int(rng.choice(len(population), p=probabilities))]

        if rng.random() < config.crossover_rate:
            parent_b = population[int(rng.choice(len(population), p=probabilities))]
            genome = NeuralGenome.crossover(parent_a.genome, parent_b.genome, rng)
        else:
            genome = parent_a.genome.copy()

        genome.mutate(rng, config.mutation_rate, config.mutation_scale)
        child = Agent(
            genome=genome,
            position=random_position(world, rng),
            energy=config.starting_energy,
            initial_energy=config.starting_energy,
        )
        child.initialize(world)
        new_population.append(child)

    return new_population


def simulate(config: Config):
    rng = np.random.default_rng(config.simulation_seed)
    world = World(generate_base_terrain(config.terrain_size, config.terrain_seed), config)
    population = create_population(world, config, rng)

    history: Dict[str, list] = {
        "best_height": [],
        "mean_height": [],
        "mean_energy": [],
        "improvements": [],
        "max_built_level": [],
    }

    best_agent = None
    best_score = -np.inf

    for generation in range(config.generations):
        for turn in range(config.turns_per_generation):
            for index in rng.permutation(len(population)):
                agent = population[index]
                if agent.alive:
                    action = agent.choose_action(
                        world,
                        turn,
                        config.turns_per_generation,
                        rng,
                        config.exploration_temperature,
                    )
                    agent.act(action, world)

        scores = fertility(population, world, config)
        heights = np.array([agent.max_height for agent in population])
        energies = np.array([agent.energy for agent in population])
        improvements = np.array([agent.successful_improvements for agent in population])

        current_best = int(np.argmax(scores))
        if scores[current_best] > best_score:
            best_score = float(scores[current_best])
            source = population[current_best]
            best_agent = Agent(
                source.genome.copy(),
                source.position.copy(),
                source.energy,
                source.initial_energy,
                source.alive,
                source.max_height,
                source.successful_improvements,
                source.successful_moves,
                list(source.path),
            )

        history["best_height"].append(float(heights.max()))
        history["mean_height"].append(float(heights.mean()))
        history["mean_energy"].append(float(energies.mean()))
        history["improvements"].append(int(improvements.sum()))
        history["max_built_level"].append(int(world.built.max()))

        if generation % config.log_every == 0 or generation == config.generations - 1:
            print(
                f"Generation {generation:3d} | "
                f"best height={heights.max():6.2f} | "
                f"mean height={heights.mean():6.2f} | "
                f"mean energy={energies.mean():6.2f} | "
                f"improvements={improvements.sum():4d} | "
                f"max built level={world.built.max():2d}"
            )

        population = reproduce(population, world, config, rng)

    return world, history, best_agent


def visualize(world: World, history: Dict[str, list], best_agent: Agent) -> None:
    plt.figure(figsize=(11, 8))
    plt.imshow(world.height, origin="lower", cmap="terrain")
    if best_agent and best_agent.path:
        path = np.array(best_agent.path)
        plt.plot(path[:, 1], path[:, 0], linewidth=2, label="Best agent path")
        plt.scatter(path[0, 1], path[0, 0], s=70, label="Start")
        plt.scatter(path[-1, 1], path[-1, 0], s=90, marker="X", label="End")
    plt.colorbar(label="Effective height")
    plt.title("Final terrain after technological modification")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 8))
    plt.imshow(world.built, origin="lower")
    plt.colorbar(label="Built level")
    plt.title("Accumulated technological infrastructure")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()

    plt.figure(figsize=(11, 5))
    plt.plot(history["best_height"], label="Best height")
    plt.plot(history["mean_height"], label="Mean height")
    plt.xlabel("Generation")
    plt.ylabel("Height")
    plt.title("Evolution of climbing performance")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(11, 5))
    plt.plot(history["mean_energy"], label="Mean remaining energy")
    plt.plot(history["improvements"], label="Improvements per generation")
    plt.plot(history["max_built_level"], label="Maximum built level")
    plt.xlabel("Generation")
    plt.title("Energy and technological investment")
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neuroevolution on a modifiable 3D terrain.")
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--population", type=int, default=220)
    parser.add_argument("--turns", type=int, default=55)
    parser.add_argument("--terrain-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        generations=args.generations,
        population_size=args.population,
        turns_per_generation=args.turns,
        terrain_size=args.terrain_size,
        simulation_seed=args.seed,
    )

    world, history, best_agent = simulate(config)

    print("\nBest historical agent")
    print(f"Maximum height:      {best_agent.max_height:.3f}")
    print(f"Remaining energy:    {best_agent.energy:.3f}")
    print(f"Improvements made:   {best_agent.successful_improvements}")
    print(f"Successful moves:    {best_agent.successful_moves}")

    visualize(world, history, best_agent)


if __name__ == "__main__":
    main()
