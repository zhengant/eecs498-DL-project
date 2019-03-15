"""
Functions for distilling policies from multiple models into a single model
"""

def generate_observations(models):
    # generate a dataset of observations from the input models to be used as training data for the distilled model
    # we might be able to get this data somehow from the training process, but that might also be a bad idea
    pass


def direct_distillation(input_models, target_model):
    # just train some other model directly on the observations from the input models
    # maybe pass in observations instead of models
    pass


def model_merge_distillation(input_models, k, env):
    # distill k models into a smaller network with a 1/k fraction of the parameters, then make a larger network that
    # concatenates the outputs of the smaller models into a few fc layers, train that resulting model on the env using
    # some RL model
    pass