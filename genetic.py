"""
Do the genetic algorithm training
"""

import numpy as np

from models import *
from environments import *

from parallel_model_training import *
from single_model_training import *

from policy_distillation import *


"""
Selects the k-best models according to fitness scores
"""
def select_k_best(k, models, fitness_scores):
    best_k_indices = np.argmax(fitness_scores)
    return models[best_k_indices]


"""
Computes the average reward for a model
Comment: Unsure if this should be recorded during training or some other way
"""
def average_reward(model):
    pass


def policy_distillation_cross_over(parent_models, model_out):
    direct_distillation(parent_models, model_out)


"""
Adds a Normal Variable to model weights with probability 1-p
"""
def mutate_network_weights(model, p=0.95):
    model_weights = None
    for i, weight in enumerate(model_weights):
        if np.random.random() > p:
            model_weights[i] = weight + np.random.normal()


class NN_GA:
    def __init__(self, generation_size, model_network, k=None, keep_parents=True, num_parents=2, select_k_fn=select_k_best, fitness_fn=average_reward, cross_over_fn=None, mutation_fn=None):
        self.models = []
        self.generation_size = generation_size
        self.template_model = model_network.copy()
        for i in range(generation_size):
            self.models.append(model_network.copy())

        if k is not None:
            self.k = max([generation_size/5, 2])

        self.keep_parents=keep_parents
        self.num_parents = num_parents
        self.select_k_fn = select_k_fn
        self.fitness_fn = fitness_fn

        self.cross_over_fn = cross_over_fn
        self.mutation_fn = mutation_fn

        self.fitness_scores = []

    def train_models(self):
        pass

    def get_models(self):
        return self.models

    def fitness_scores(self):
        self.fitness_scores = []
        for model in self.models:
            self.fitness_scores.append(self.fitness_fn(model))

    def get_fitness_scores(self):
        return self.fitness_scores

    """
    Forms new generation using the k-best parents.
    Parents can be included in the new generation.
    """
    def next_generation(self):
        self.fitness_scores()
        parents = self.select_k_fn(self.k, self.models, self.fitness_scores)

        self.models = []
        if self.keep_parents:
            self.models = parents

        while len(self.models) < self.generation_size:
            child_model = self.template_model.copy()
            if self.cross_over_fn is not None:
                current_parents = np.random.choice(parents, size=self.num_parents, replace=False)
                self.cross_over_fn(current_parents, child_model)

            if self.mutation_fn is not None:
                self.mutation_fn(child_model)

            self.models.append(child_model)