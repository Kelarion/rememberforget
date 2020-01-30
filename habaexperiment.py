"""
Remember-Forget experiments

Take sequences of tokens drawn from an alphabet, return at the end of the sequence
the token that wasn't repeated. This script is for sending to the cluster.

Command-line arguments:
    -v [--verbose]  set verbose
    -n val          number of neurons (int)
    -p val          size of alphabet (int)
    -q val          size of subalphabet (int)
    -l val          number of tokens in sequence (int)
    -s              flag to save the train/test sets
"""

#%%
import socket
import os

if socket.gethostname() == 'kelarion':
    CODE_DIR = '/home/matteo/Documents/github/rememberforget/'
    SAVE_DIR = '/home/matteo/Documents/github/rememberforget/'
else:    
    CODE_DIR = '/rigel/home/ma3811/remember-forget/'
    SAVE_DIR = '/rigel/theory/users/ma3811/'
    
import getopt, sys
sys.path.append(CODE_DIR)

import math
import pickle
import torch
import torch.optim as optim
import numpy as np
from students import RNNModel
from needful_functions import *

#%% parse arguments
allargs = sys.argv

arglist = allargs[1:]

unixOpts = "vsn:p:l:q:"
gnuOpts = ["verbose"]

opts, _ = getopt.getopt(arglist, unixOpts, gnuOpts)

if len(opts) == 0:
    sys.exit('No arguments supplied! I`m guessing you wanted the help text: \n'
             '\n'
             'Command-line arguments: \n'
             ' -v, --verbose   set verbose \n'
             ' -n val          number of neurons (int) \n'
             ' -p val          size of alphabet (int) \n'
             ' -l val          number of tokens in sequence (int) \n'
             ' -q val          size of subalphabet (int); default None \n'
             ' -s              flag to save the train/test sets; default False \n')

verbose, save_sets, N, P, L, Q = False, False, 10, 10, 5, None # defaults
for op, val in opts:
    if op in ('-v','--verbose'):
        verbose = True
    if op in ('-s'):
        save_sets = True
    if op in ('-n'):
        try:
            N = int(val)
        except ValueError:
            N = float(val)
    if op in ('-p'):
        P = int(val)
    if op in ('-q'):
        Q = int(val)
    if op in ('-l'):
        if ',' in val:
            L = val.split(',')
            L = [int(l) for l in L]
        else:
            L = [int(val)]

if type(N) is float:
    N = int(N*P)

#%% run experiment
# sequence parameters
forget = False  # does the task have a forget component?
explicit = False
be_picky = False        # should we only use unique data?
nseq = 5000
ntest = 500
pad = -1
dead_time = None           # amount of time witout any input
# RNN parameters
rnn_type = 'tanh'
persistent = False      # encode context as a persistent cue?
embed = False          # use trainable embedding (T) or indicator (F)
nlayers = 1            # number of recurrent networks

# optimisation parameters
nepoch = 2000               # max how many times to go over data
alg = optim.Adam
dlargs = {'num_workers': 2, 
          'batch_size': 64, 
          'shuffle': True}  # dataloader arguments
optargs = {'lr': 5e-4}      # optimiser args
if forget:
    criterion = torch.nn.CrossEntropyLoss()
else:
    if Q is None:
        w = torch.ones(P)
    else:
        w = torch.zeros(P)
        w[:Q] = 1
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum',weight=w)

# organise the low-level arguments, i.e. things passed down to `draw_seqs`
if forget:
    if Q is not None:
        sig_toks = list(range(Q))
    else:
        sig_toks = None
else:
    sig_toks = None
    
if not forget: # overwrite these 
    explicit = False
    persistent = False
    dead_time = None

expargs = {'forget': forget,
           'explicit': explicit,
           'be_picky': be_picky,
           'signal_tokens': sig_toks}  # possible output tokens

# package (for saving later?)
fixed_args = {'rnn_type': rnn_type,
              'nseq': nseq,
              'ntest': ntest,
              'nlayers': nlayers,
              'pad': pad,
              'dlargs': dlargs,
              'optargs': optargs,
              'criterion': criterion,
              'expargs': expargs,
              'alg': alg,
              'nepoch': nepoch,
              'dead_time': dead_time,
              'train_embedding': embed,
              'persistent': persistent,
              'verbose': verbose,
              'return_data': save_sets} # why do I do this, actually?

print('Starting with N=%d, P=%d ...'%(N, P))
if save_sets:
    accuracy, rnn, D = train_and_test(N=N, P=P, Ls=L, **fixed_args)
else:
    accuracy, rnn = train_and_test(N=N, P=P, Ls=L, **fixed_args)

# save results
FOLDERS = expfolders(explicit, persistent, embed, forget)
expinf = suffix_maker(N, P, L, rnn_type, Q, dead_time)

if not os.path.isdir(SAVE_DIR+FOLDERS):
    os.makedirs(SAVE_DIR+FOLDERS)

params_fname = 'parameters'+expinf+'.pt'
#loss_fname = 'loss'+expinf+'.npy'
#tloss_fname = 'test_loss'+expinf+'.npy'
metrics_fname = 'metrics'+expinf+'.pkl'
args_fname = 'arguments'+expinf+'.npy'
accy_fname = 'accuracy'+expinf+'.npy'

rnn.save(SAVE_DIR+FOLDERS+params_fname)
#with open(SAVE_DIR+FOLDERS+loss_fname, 'wb') as f:
#    np.save(f, rnn.train_loss)
#with open(SAVE_DIR+FOLDERS+tloss_fname, 'wb') as f:
#    np.save(f, rnn.test_loss)
with open(SAVE_DIR+FOLDERS+metrics_fname, 'wb') as f:
    pickle.dump(rnn.metrics, f, -1)
with open(SAVE_DIR+FOLDERS+accy_fname, 'wb') as f:
    np.save(f, accuracy)
with open(SAVE_DIR+FOLDERS+args_fname, 'wb') as f:
    fixed_args['Ls'] = L
    np.save(f, fixed_args)

if save_sets:
    dset_fname = 'datasets'+expinf+'.pkl'
    with open(SAVE_DIR+FOLDERS+dset_fname, 'wb') as f:
        pickle.dump(D, f, -1)

print('done')    
print(':' + ')'*12)


