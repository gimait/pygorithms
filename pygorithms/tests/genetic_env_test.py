"""
    Tests for genetic algorithms
"""
import logging
import sys
import time
import unittest

import numpy as np

from pygorithms.genetic_env import GeneticEnvironment


def sig(x):
    """ Activation function for NN. """
    return 1 / (1 + np.exp(-x))


def logic_operator_network(i1, i2, params):
    """ Network to test training. """
    n1 = sig(params[0] + i1 * params[1] + i2 * params[2])
    n2 = sig(params[3] + i1 * params[4] + i2 * params[5])
    out = sig(params[6] + n1 * params[7] + n2 * params[8])
    return out


def OR(params):
    """ OR target function. """
    error = 0
    input_output = [((0, 0), 0),
                    ((0, 1), 1),
                    ((1, 0), 1),
                    ((1, 1), 1)]
    for (i1, i2), o in input_output:
        error += abs(logic_operator_network(i1, i2, params) - o)
    return error


def XOR(params):
    """ XOR target function. """

    error = 0
    input_output = [((0, 0), 0),
                    ((0, 1), 1),
                    ((1, 0), 1),
                    ((1, 1), 0)]
    for (i1, i2), o in input_output:
        error += abs(logic_operator_network(i1, i2, params) - o)
    return error


def AND(params):
    """ AND target function. """
    error = 0
    input_output = [((0, 0), 1),
                    ((0, 1), 0),
                    ((1, 0), 0),
                    ((1, 1), 1)]
    for (i1, i2), o in input_output:
        error += abs(logic_operator_network(i1, i2, params) - o)
    return error


def XAND(params):
    """ XAND target function. """
    error = 0
    input_output = [((0, 0), 0),
                    ((0, 1), 0),
                    ((1, 0), 0),
                    ((1, 1), 1)]
    for (i1, i2), o in input_output:
        error += abs(logic_operator_network(i1, i2, params) - o)
    return error


class TestGeneticIntegration(unittest.TestCase):
    """
        Integration tests for Genetic Environment.
    """
    def setUp(self):
        self.log = logging.getLogger("TestLog")

    def test_OR(self):
        bench = GeneticEnvironment(chromosome_size=9,
                                   population=1000,
                                   error_function=OR,
                                   survival_rate=0.2,
                                   mutation_rate=0.5,
                                   value_range=(10, -10),
                                   hard_mutation_rate=0.5,
                                   radical_rate=0.1)

        t0 = time.perf_counter()
        sol, err = bench.run_optimization()
        dt = time.perf_counter() - t0
        self.log.debug('OR training time: {}'.format(dt))
        self.assertTrue(dt < 30)
        self.assertAlmostEqual(err, 0.0, places=4)

    def test_XOR(self):
        bench = GeneticEnvironment(chromosome_size=9,
                                   population=1000,
                                   error_function=XOR,
                                   survival_rate=0.2,
                                   mutation_rate=0.5,
                                   value_range=(10, -10),
                                   hard_mutation_rate=0.5,
                                   radical_rate=0.1)

        t0 = time.perf_counter()
        sol, err = bench.run_optimization()
        dt = time.perf_counter() - t0
        self.log.debug('XOR training time: {}'.format(dt))
        self.assertTrue(dt < 30)
        self.assertAlmostEqual(err, 0.0, places=4)

    def test_AND(self):
        bench = GeneticEnvironment(chromosome_size=9,
                                   population=1000,
                                   error_function=AND,
                                   survival_rate=0.2,
                                   mutation_rate=0.5,
                                   value_range=(10, -10),
                                   hard_mutation_rate=0.5,
                                   radical_rate=0.1)

        t0 = time.perf_counter()
        sol, err = bench.run_optimization()
        dt = time.perf_counter() - t0
        self.log.debug('AND training time: {}'.format(dt))
        self.assertTrue(dt < 30)
        self.assertAlmostEqual(err, 0.0, places=4)

    def test_XAND(self):
        bench = GeneticEnvironment(chromosome_size=9,
                                   population=1000,
                                   error_function=XAND,
                                   survival_rate=0.2,
                                   mutation_rate=0.5,
                                   value_range=(10, -10),
                                   hard_mutation_rate=0.5,
                                   radical_rate=0.1)

        t0 = time.perf_counter()
        sol, err = bench.run_optimization()
        dt = time.perf_counter() - t0
        self.log.debug('XAND training time: {}'.format(dt))
        self.assertTrue(dt < 30)
        self.assertAlmostEqual(err, 0.0, places=4)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
