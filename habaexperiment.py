"""
Remember-Forget experiments

Take sequences of tokens drawn from an alphabet, return at the end of the sequence
the token that wasn't repeated. This one is for sending to the cluster.

Bash arguments:
    -v [--verbose]  set verbose
    -n val          number of neurons (int)
    -p val          size of alphabet (int)
    -l val          number of tokens (int)
"""

#%%
#CODE_DIR = r'/rigel/home/ma3811/remember-forget/'
CODE_DIR = r'/home/matteo/Documents/github/rememberforget/'

import getopt, sys
sys.path.append(CODE_DIR)

import torch
import torch.optim as optim
import numpy as np
from students import RNNModel, stateDecoder
from needful_functions import *

#%% parse arguments
allargs = sys.argv

arglist = allargs[1:]

unixOpts = "vn:p:l:"
gnuOpts = ["verbose"]

opts, _ = getopt.getopt(arglist, unixOpts, gnuOpts)

verbose, N, P, L = False, 10, 10, 5 # defaults
for op, val in opts:
    if op in ('-v','--verbose'):
        verbose = True
    if op in ('-n'):
        try:
            N = int(val)
        except ValueError:
            N = float(val)
    if op in ('-p'):
        P = int(val)
    if op in ('-l'):
        if ',' in val:
            L = val.split(',')
            L = [int(l) for l in L]
        else:
            L = int(val)

if type(N) is float:
    N = int(N*P)

#%% run experiment
# parameters
explicit = True  

nlayers = 1                 # number of recurrent networks
rnn_type = 'GRU'
be_picky = True             # should we only use unique data?

nseq = 5000
ntest = 500
pad = -1

# optimisation parameters
nepoch = 2000               # max how many times to go over data
alg = optim.Adam
dlargs = {'num_workers': 2, 
          'batch_size': 64, 
          'shuffle': True}  # dataloader arguments
optargs = {'lr': 5e-4}      # optimiser args
criterion = torch.nn.NLLLoss()


# organise arguments for multiprocessing
fixed_args = {'rnn_type': rnn_type,
              'explicit': explicit,
              'nseq': nseq,
              'ntest': ntest,
              'nlayers': nlayers,
              'pad': pad,
              'dlargs': dlargs,
              'optargs': optargs,
              'alg': alg,
              'nepoch': nepoch,
              'be_picky': be_picky,
              'verbose': verbose} # all the arguments we don't iterate over

#iter_args = product(Ns,Ps,Ls)

print('Starting with N=%d, P=%d ...'%(N, P))
loss, accuracy, rnn = train_and_test(N=N, P=P, Ls=L, **fixed_args)
#num_cores = multiprocessing.cpu_count()
#parfor = Parallel(n_jobs=num_cores, verbose=5)
#
#out = parfor(delayed(train_and_test)(N=n, AB=ab, Ls=l, 
#                                     **fixed_args) for n,ab,l in iter_args)

# save results
params_fname = 'parameters_%d_%d_%s.pt'%(N, P, rnn_type)
loss_fname = 'loss_%d_%d_%s.npy'%(N, P, rnn_type)
accy_fname = 'accuracy_%d_%d_%s.npy'%(N, P, rnn_type)
rnn_specs_fname = 'rnn_specs_%d_%d_%s.npy'%(N, P, rnn_type)

rnn.save(CODE_DIR+'results/'+params_fname)
with open(CODE_DIR+'results/' +loss_fname, 'wb') as f:
    np.save(f, loss)
with open(CODE_DIR+'results/'+accy_fname, 'wb') as f:
    np.save(f, accuracy)
#with open(CODE_DIR+'results/'+rnn_specs_fname, 'wb') as f:
#    np.save(f, accuracy)


print('done')    
print(':' + ')'*12)


