"""
Collection of environment objects to train on. Probably want these objects to match the openai gym API

We can probably wrap a bunch of openai gym environments with our own stuff to create custom reward functions/outputs

Depending on how many different environments we have/how complicated they get we may want to split these into separate
files
"""

class BasicGridWorld:
    # maybe want separate classes for the distillation environment and the genetic algorithm environment
    def __init__(self):
        pass
    def step(self):
        pass