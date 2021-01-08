"""
    Methods for genetic algorithms
"""

import random
from multiprocessing.pool import ThreadPool
from typing import Callable, List, Optional, Tuple

import numpy as np


class GeneticEnvironment:
    """
        Environment to train models using genetic algorithms.
    """

    def __init__(self,
                 chromosome_size: int,
                 population: int,
                 error_function: Callable,
                 survival_rate: float,
                 mutation_rate: float,
                 hard_mutation_rate: Optional[float] = 0.0,
                 radical_rate: Optional[float] = 0.0,
                 value_range: Optional[Tuple[float, float]] = None,
                 max_iterations: Optional[int] = 10000,
                 max_stale_iterations: Optional[int] = 200) -> None:

        self.error_function = error_function
        self.population = [np.zeros(chromosome_size) for _ in range(population)]

        # This is the number of elements that are used to generate the new population
        self.survivor_number = int(survival_rate * population)
        if self.survivor_number % 2 != 0:
            self.survivor_number += 1
        self.mutation_rate = mutation_rate
        self.hard_mutation_rate = hard_mutation_rate
        self.radical_rate = radical_rate

        self.value_range = value_range if value_range is not None else (-1, 1)
        self.max_iterations = max_iterations
        self.max_stale_iterations = max_stale_iterations

    def reset_population(self) -> None:
        for individual in self.population:
            for i in range(individual.size):
                individual[i] = random.uniform(self.value_range[0], self.value_range[1])

    def run_optimization(self) -> Tuple[np.array, float]:
        self.reset_population()
        last_min_error = 1000
        it = 0
        st = 0

        def calc_error(enum):
            (i, individual) = enum
            return self.error_function(individual), i

        while True:
            pool = ThreadPool(10)
            errors = pool.map(calc_error, enumerate(self.population))
            pool.close()
            pool.join()

            errors.sort()

            last_min_error = errors[0][0]

            # Check if we should stop
            if it > self.max_iterations or st > self.max_stale_iterations:
                break
            if errors[0][0] >= last_min_error:
                st += 1
            else:
                st = 0
            it += 1

            # Discard top % and repopulate.
            to_repopulate = [self.population[idx] for err, idx in errors[0:self.survivor_number]]
            self.generation_transit(to_repopulate)

        # Return the found solution
        return self.population[0], last_min_error

    def generation_transit(self, genomes: List[np.array]) -> None:
        # Keep always the best one
        self.population[0] = genomes[0]
        # Make two groups
        random.shuffle(genomes)
        g1 = genomes[0:int(len(genomes) / 2)]
        g2 = genomes[int(len(genomes) / 2):]
        assert(len(g1) == len(g2))

        # Create new population
        for i in range(1, len(self.population)):
            # Phase 1: Mix-in
            subgroup_idx = i % len(g1)
            self.population[i] = self.mix_genomes(g1[subgroup_idx], g2[subgroup_idx])
            # Phase 2: soft mutation
            self.population[i] = self.apply_mutation(self.population[i])
        # Phase 3: hard mutation, add a small number of radicals to shake the population
        for i in range(1, int(len(self.population) * self.radical_rate)):
            self.population[-i] = self.apply_hard_mutation(self.population[-i])

        return

    @staticmethod
    def mix_genomes(gen1: np.array, gen2: np.array) -> np.array:
        child = np.zeros(gen1.shape)
        for i in range(child.size):
            child[i] = random.choice((gen1[i], gen2[i], (gen1[i] + gen2[i]) / 2))
        return child

    def apply_mutation(self, gen: np.array) -> np.array:
        for i in range(gen.size):
            if random.random() < self.mutation_rate:
                gen[i] += random.uniform(self.value_range[0], self.value_range[1]) * 0.1
        return gen

    def apply_hard_mutation(self, gen: np.array) -> np.array:
        for i in range(gen.size):
            if random.random() < self.hard_mutation_rate:
                gen[i] = random.uniform(self.value_range[0], self.value_range[1])
        return gen
