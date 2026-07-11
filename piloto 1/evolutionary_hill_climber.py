import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# ============================================================
# TERRAIN
# ============================================================

def generate_terrain(
    size=150,
    large_smoothing=18,
    medium_smoothing=7,
    seed=42,
):
    """Generate a continuous, irregular 2D terrain represented by heights."""
    rng = np.random.default_rng(seed)

    noise_1 = rng.normal(size=(size, size))
    noise_2 = rng.normal(size=(size, size))
    noise_3 = rng.normal(size=(size, size))

    large_layer = gaussian_filter(noise_1, sigma=large_smoothing)
    medium_layer = gaussian_filter(noise_2, sigma=medium_smoothing)
    small_layer = gaussian_filter(noise_3, sigma=2)

    large_layer /= np.std(large_layer)
    medium_layer /= np.std(medium_layer)
    small_layer /= np.std(small_layer)

    terrain = (
        8.0 * large_layer
        + 3.0 * medium_layer
        + 0.6 * small_layer
    )

    # Spatial variance: some areas are more mountainous than others.
    y = np.linspace(0, 1, size)
    variance_map = 0.45 + 1.1 * y[:, None]

    terrain *= variance_map
    terrain -= terrain.min()

    return terrain


# ============================================================
# INDIVIDUAL AGENT
# ============================================================

class Individual:
    """
    Genome values:

    genome[0] = preference for climbing
    genome[1] = directional persistence
    genome[2] = random exploration
    genome[3] = preference for steep or gentle slopes
    """

    def __init__(self, genome, position):
        self.genome = np.array(genome, dtype=float)
        self.position = np.array(position, dtype=int)
        self.previous_direction = np.array([0, 0], dtype=int)
        self.path = [tuple(self.position)]

    def choose_move(self, terrain, rng):
        height, width = terrain.shape
        x, y = self.position

        moves = [
            np.array([-1, -1]),
            np.array([-1, 0]),
            np.array([-1, 1]),
            np.array([0, -1]),
            np.array([0, 1]),
            np.array([1, -1]),
            np.array([1, 0]),
            np.array([1, 1]),
        ]

        current_height = terrain[x, y]
        scores = []

        climbing = self.genome[0]
        persistence = self.genome[1]
        exploration = self.genome[2]
        slope_sensitivity = self.genome[3]

        for move in moves:
            new_position = self.position + move
            nx, ny = new_position

            if nx < 0 or nx >= height or ny < 0 or ny >= width:
                scores.append(-np.inf)
                continue

            new_height = terrain[nx, ny]
            height_difference = new_height - current_height

            climbing_score = climbing * height_difference
            persistence_score = persistence * np.dot(
                move,
                self.previous_direction,
            )
            slope_score = slope_sensitivity * abs(height_difference)
            random_score = rng.normal(0, max(exploration, 0.001))

            scores.append(
                climbing_score
                + persistence_score
                + slope_score
                + random_score
            )

        return moves[int(np.argmax(scores))]

    def move(self, terrain, rng):
        movement = self.choose_move(terrain, rng)
        self.position += movement
        self.previous_direction = movement
        self.path.append(tuple(self.position))

    def fitness(self, terrain):
        x, y = self.position
        return terrain[x, y]


# ============================================================
# EVOLUTION
# ============================================================

def create_population(amount, terrain, rng):
    population = []
    height, width = terrain.shape

    for _ in range(amount):
        genome = np.array([
            rng.uniform(0, 3),
            rng.uniform(-1, 2),
            rng.uniform(0.05, 2),
            rng.uniform(-1, 1),
        ])

        position = [
            rng.integers(0, height),
            rng.integers(0, width),
        ]

        population.append(Individual(genome, position))

    return population


def reproduce(
    population,
    terrain,
    rng,
    offspring_amount,
    mutation_rate=0.15,
):
    fitness_values = np.array([
        individual.fitness(terrain)
        for individual in population
    ])

    fitness_values -= fitness_values.min()
    fitness_values += 0.001

    probabilities = fitness_values ** 2
    probabilities /= probabilities.sum()

    new_population = []
    height, width = terrain.shape

    for _ in range(offspring_amount):
        parent_index = rng.choice(len(population), p=probabilities)
        parent = population[parent_index]

        mutation = rng.normal(
            0,
            mutation_rate,
            size=len(parent.genome),
        )

        new_genome = parent.genome + mutation
        new_genome[2] = max(new_genome[2], 0.001)

        position = [
            rng.integers(0, height),
            rng.integers(0, width),
        ]

        new_population.append(Individual(new_genome, position))

    return new_population


def simulate_evolution(
    terrain,
    generations=100,
    population_size=250,
    moves_per_generation=40,
    mutation_rate=0.12,
    seed=20,
):
    rng = np.random.default_rng(seed)
    population = create_population(population_size, terrain, rng)

    best_history = []
    average_history = []
    best_individual_ever = None
    best_fitness_ever = -np.inf

    for generation in range(generations):
        for individual in population:
            for _ in range(moves_per_generation):
                individual.move(terrain, rng)

        fitness_values = np.array([
            individual.fitness(terrain)
            for individual in population
        ])

        best_index = int(np.argmax(fitness_values))
        best_individual = population[best_index]
        best_fitness = fitness_values[best_index]

        best_history.append(best_fitness)
        average_history.append(np.mean(fitness_values))

        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            best_individual_ever = best_individual

        if generation % 10 == 0:
            print(
                f"Generation {generation:3d} | "
                f"Best height: {best_fitness:7.2f} | "
                f"Average height: {np.mean(fitness_values):7.2f}"
            )

        population = reproduce(
            population,
            terrain,
            rng,
            population_size,
            mutation_rate,
        )

    return {
        "final_population": population,
        "best_history": best_history,
        "average_history": average_history,
        "best_individual": best_individual_ever,
        "best_fitness": best_fitness_ever,
    }


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_results(terrain, results):
    best = results["best_individual"]

    plt.figure(figsize=(10, 8))
    plt.imshow(terrain, origin="lower", cmap="terrain")

    path = np.array(best.path)

    plt.plot(
        path[:, 1],
        path[:, 0],
        linewidth=2,
        label="Best individual's path",
    )
    plt.scatter(
        path[0, 1],
        path[0, 0],
        s=70,
        label="Start",
    )
    plt.scatter(
        path[-1, 1],
        path[-1, 0],
        s=90,
        marker="X",
        label="End",
    )

    plt.colorbar(label="Height")
    plt.title(f"Best individual: height {results['best_fitness']:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(results["best_history"], label="Best height")
    plt.plot(results["average_history"], label="Average height")
    plt.xlabel("Generation")
    plt.ylabel("Height reached")
    plt.title("Population evolution")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    terrain = generate_terrain(size=150, seed=42)

    results = simulate_evolution(
        terrain=terrain,
        generations=120,
        population_size=300,
        moves_per_generation=45,
        mutation_rate=0.10,
        seed=10,
    )

    best = results["best_individual"]

    print("\nBest genome found:")
    print(f"Climbing preference: {best.genome[0]:.3f}")
    print(f"Persistence:         {best.genome[1]:.3f}")
    print(f"Exploration:         {best.genome[2]:.3f}")
    print(f"Slope sensitivity:   {best.genome[3]:.3f}")

    visualize_results(terrain, results)
