import os
import subprocess
import time
import threading
import shlex
import queue

import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U
from baselines.deepq.utils import ObservationInput
from reversi_environment import legal_moves
from models import reversi_network
from baselines.common.tf_util import save_variables, load_variables, get_session


def load_model(model_name, env):
    num_layers = 3
    num_hidden = 128
    activation = tf.nn.relu
    layer_norm = False

    model_load_path = "".join(['saved_models/', model_name, '.pkl'])
    model = PredictorAgent(reversi_network(num_layers=num_layers,
                                          num_hidden=num_hidden,
                                          activation=activation,
                                          layer_norm=layer_norm),
                        env,
                        model_scope=model_name)
    load_variables(model_load_path, variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name))
    return model

class AbstractAgent():
    def update(self, *args, **kwargs):
        pass

    def predict(self, board):
        pass

    def input_opp_move(self, row, col):
        pass

    def reset(self):
        pass


class RandomAgent(AbstractAgent):
    def update(self, *args, **kwargs):
        pass

    def predict(self, board):
        board_size = board.shape[0]
        return np.random.rand(board_size, board_size)


class PlayerAgent(AbstractAgent):
    def update(self, *args, **kwargs):
        pass

    def predict(self, board):
        print(board[:,:,1] - board[:,:,2])
        q_vals = np.zeros((8,8))

        legal_moves_mask = legal_moves(board, 1, 8).astype(bool)

        row = int(input("Select a row: "))
        col = int(input("Select a col: "))

        while not legal_moves_mask[row][col]:
            print("Illegal Move, please select a legal move!")
            row = int(input("Select a row: "))
            col = int(input("Select a col: "))

        q_vals[row][col] = 1
        return q_vals


class ModelLoader(AbstractAgent):
    def __init__(self, env, model_scope, other_model_names):
        num_other_models = len(other_model_names)

        self.other_models = []
        for other_model_name in other_model_names:
            model = load_model(other_model_name, env)
            self.other_models.append(model)

    def update(self, network_to_clone_vars):
        pass

    def predict(self, board):
        q_vals_all = []
        for model in self.other_models:
            q_vals = model.predict(np.expand_dims(board, axis=0)).reshape((8,8))
            q_vals_all.append(q_vals)
        q_vals_all = np.stack(q_vals_all, axis=-1)

        board = np.concatenate([board[:,:,0:3], q_vals_all, np.expand_dims(board[:,:,-1], axis=-1)], axis=-1)
        return board


class PredictorAgent(AbstractAgent):
    def __init__(self, network, env, model_scope):
        with tf.variable_scope(model_scope, reuse=False):
            observation_space = env.observation_space
            inputs = ObservationInput(observation_space)
            outputs = network(inputs.get())
            self.network = U.function([inputs], outputs)
            self.model_scope = model_scope

    def update(self, network_to_clone_vars):
        pass

    def predict(self, board):
        return self.network(np.expand_dims(board, axis=0))


class CompositeAgent(AbstractAgent):
    def __init__(self, network, env, model_scope, other_models):
        num_other_models = len(other_models)
        self.other_models = other_models
        self.model_scope = model_scope

        with tf.variable_scope(self.model_scope, reuse=False):
            observation_space = env.observation_space
            observation_space.shape = (8, 8, 4+num_other_models)
            inputs = ObservationInput(observation_space)
            outputs = network(inputs.get())
            self.network = U.function([inputs], outputs)

    def update(self, network_to_clone_vars):
        pass

    def predict(self, board):
        if not len(self.other_models) == 0:
            q_vals_all = []
            for model in self.other_models:
                q_vals = model.predict(np.expand_dims(board, axis=0)).reshape((8, 8))
                q_vals_all.append(q_vals)
            q_vals_all = np.stack(q_vals_all, axis=-1)
            board = np.concatenate([board[:,:,0:3], q_vals_all, np.expand_dims(board[:,:,-1], axis=-1)], axis=-1)


        return self.network(np.expand_dims(board, axis=0))


class ClonedAgent(AbstractAgent):
    def __init__(self, network, env, model_scope="cloned_agent", other_models=[]):
        with tf.variable_scope(model_scope, reuse=False):
            observation_space = env.observation_space
            inputs = ObservationInput(observation_space)
            outputs = network(inputs.get())
            self.network = U.function([inputs], outputs)
            self.model_scope = model_scope
            self.other_models = other_models

    def update(self, network_to_clone_vars):
        this_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope)
        sess = U.get_session()
        sess.run([v_this.assign(v) for v_this, v in zip(this_vars, network_to_clone_vars)])

    def predict(self, board):
        if not len(self.other_models) == 0:
            q_vals_all = []
            for model in self.other_models:
                q_vals = model.predict(np.expand_dims(board, axis=0)).reshape((8, 8))
                q_vals_all.append(q_vals)
            q_vals_all = np.stack(q_vals_all, axis=-1)
            board = np.concatenate([board[:,:,0:3], q_vals_all, np.expand_dims(board[:,:,-1], axis=-1)], axis=-1)


        return self.network(np.expand_dims(board, axis=0))


class EdaxAgent(AbstractAgent):
    def __init__(self, engine_dir='edax-reversi/bin/', engine_name='lEdax-x64-modern', ply=1, num_tasks=1):
        self.engine_dir = engine_dir
        self.engine_name = engine_name
        self.ply = ply
        self.num_tasks = num_tasks
        self.engine_active = False
        self.std_start_fen = "8/8/8/3pP3/3Pp3/8/8/8 b - - 0 1"
        self.p = None
        self.p_out = queue.Queue()
        self.end_stdout = False
        self.go = False

        self.engine_init()

    # def __del__(self):
    #     if self.p.poll() is not None:
    #         self.p.kill()

    #     self.end_stdout = True
    #     self.soutt.join()
    
    def reset(self):
        if self.soutt is not None:
            self.end_stdout = True
            self.soutt.join()
            self.p_out = queue.Queue()
            self.end_stdout = False
            self.soutt = threading.Thread(target=self.read_stdout).start()
        self.command('force\n')
        self.command("setboard " + self.std_start_fen + "\n")
        self.go = False

    def input_opp_move(self, row, col):
        if type(col) is not str:
            col = "abcdefgh"[col]
        row = row + 1
        move = col + str(row)
        self.command('usermove ' + move + '\n')

    def predict(self, board):
        if not self.go:
            self.command('go\n')
            self.go = True

        row, col = self.get_edax_move()

        q = np.zeros((8, 8))
        q[row, col] = np.inf

        return q

    def engine_init(self):
        self.engine_active = False
        if not os.path.exists(os.path.join(self.engine_dir, self.engine_name)):
            print("Error enginepath does not exist")
            return

        engine_path = os.path.join(self.engine_dir, self.engine_name)
        eval_path = os.path.join(self.engine_dir, 'data', 'eval.dat')
        book_path = os.path.join(self.engine_dir, 'data', 'book.dat')

        arglist = [engine_path,
                    "-xboard",  
                    "-n", str(self.num_tasks), 
                    '-eval-file', eval_path, 
                    '-book-file', book_path
        ]

        # engine working directory containing the executable
        # engine_wdir = os.path.dirname(self.engine_path)

        try:
            p = subprocess.Popen(arglist, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            self.p = p
        except OSError as e:
            print("Error starting engine - check path/permissions")
            print(e)
            #tkMessageBox.showinfo("OthelloTk Error", "Error starting engine",
            #                       detail="Check path/permissions")
            return

        # check process is running
        i = 0
        while (p.poll() is not None):            
            i += 1
            if i > 40:        
                print("unable to start engine process")
                return False
            time.sleep(0.25)        

        # start thread to read stdout
        self.soutt = threading.Thread(target=self.read_stdout, daemon=True).start()
        #self.command('xboard\n')
        self.command('protover 2\n')

        # Engine should respond to "protover 2" with "feature" command
        i = 0
        response = ''
        while True:            
            response = self.p_out.get() 
            if response.startswith('feature'):
                break       
            i += 1
            if i > 60:            
                print("Error - no response from engine")
                return
            time.sleep(0.25)

        self.command('variant reversi\n')
        self.command("setboard " + self.std_start_fen + "\n")
        self.command('sd ' + str(self.ply) + '\n')
        # self.command("st " + str(self.settings["time_per_move"]) + "\n") # time per move in seconds
        #self.command('sd 4\n')
        #sd = "sd " + str(self.settings["searchdepth"]) + "\n"
        #print "setting search depth:",sd
        #self.command(sd)
        self.engine_active = True        

    def command(self, cmd):
        try:
            self.p.stdin.write(bytes(cmd, "UTF-8"))
            self.p.stdin.flush()
        except AttributeError:
            print("AttributeError")
        except IOError:
            print("ioerror")

    def read_stdout(self):
        while True:
            try:
                if self.end_stdout:
                    return
                self.p.stdout.flush()
                line = self.p.stdout.readline()
                line = line.decode("UTF-8")
                line = line.strip()
                if line == '':
                    print("eof reached in read_stdout")
                    return  
                print(line)
                self.p_out.put(line)
            except Exception as e:
                print("subprocess error in read_stdout:",e)

    # convert move to board coordinates (e.g. "d6" goes to 3, 5)
    def conv_to_coord(self, mv):
        letter = mv[0]
        num = mv[1]
        col = "abcdefgh".index(letter)
        row = int(num) - 1
        return row, col

    def get_edax_move(self, s=0):
        # Check for move from engine
        line = ''
        while not line.startswith('move'):
            line = self.p_out.get()
        
        mv = line[7:]

        # convert move to board coordinates (e.g. "d6" goes to 3, 5)
        row, col = self.conv_to_coord(mv)

        return row, col
