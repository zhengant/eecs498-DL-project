"""
Functions for training many models in parallel

remember that you can only train one model on a single GPU at a time lol
"""

import threading

def train_many(envs, models, algorithms, params):
    # probably make all of these parameters lists and loop throuh all of them, starting a new thread for each
    # maybe want a list of devices to train on as well? or we could just look for available ones in this function

    # we can probably just have one function that can handle both training a bunch of models identically vs trying
    # to create sub-experts or otherwise introducing diversity in the models
    pass