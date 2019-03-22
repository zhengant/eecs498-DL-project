"""
Functions for training many models in parallel

remember that you can only train one model on a single GPU at a time lol
"""

import threading
from single_model_training import dqn

class trainer(threading.Thread):
     def __init__(self, threadID, name, run_func):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.run_func = run_func

     def run(self):
        run_func()

def train_many(envs, models, algorithms, params):
    # probably make all of these parameters lists and loop through all of them, starting a new thread for each
    # maybe want a list of devices to train on as well? or we could just look for available ones in this function

    # we can probably just have one function that can handle both training a bunch of models identically vs trying
    # to create sub-experts or otherwise introducing diversity in the models
    threads = []

    for i in range(len(models)):
        thread = trainer(i, "Agent-%i" % (i,),
            dqn(envs[i], models[i], params[i]))
        thread.start()
        threads.append(thread)
        print('Started thread: %i' % (i,))

    for t in threads:
        t.join()

if __name__ == '__main__':
    train_many([], [], [], {})
