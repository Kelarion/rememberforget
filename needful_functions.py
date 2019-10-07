import getopt, sys
#sys.path.append(CODE_DIR)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import compress, permutations
#from joblib import Parallel, delayed
#import multiprocessing
import numpy as np
import scipy.special as spc

#%%
def train_and_test(rnn_type, N, P, Ls, nseq, ntest, explicit=True,
                   nlayers=1, pad=None, dlargs=None, optargs=None,
                   criterion=None, alg=None, be_picky=False, verbose=True):
    """
    function to reduce clutter below
    """
    
    AB = list(range(P))
    nneur = N                  # size of recurrent network
    train_embedding = True      # train the embedding of numbers?
    if explicit:
        forget_switch =  max(AB)+1
    else:
        forget_switch = None
    
    ninp = len(AB)              # dimension of RNN input
    
    if type(Ls) is not list:
        Ls = [Ls]
    
    # optimisation parameters
    nepoch = 2000               # max how many times to go over data
    if alg is None:
        alg = optim.Adam
    if dlargs is None:
        dlargs = {'num_workers': 2, 
                  'batch_size': 64, 
                  'shuffle': True}  # dataloader arguments
    if optargs is None:
        optargs = {'lr': 5e-4}      # optimiser args
    if criterion is None:
        criterion = torch.nn.NLLLoss()
    
    if forget_switch is not None:
        ninp += 1
        AB_ = AB+[forget_switch] # extended alphabet
    else:
        AB_ = AB
    
    # train and test
    if verbose:
        print('TRAINING %s NETWORK' % str(Ls))
    
    nums, ans, numstest, anstest = make_dset(Ls, AB, int(nseq/len(Ls)), 
                                             ntest, 
                                             forget_switch,
                                             padding=pad)
    
    ptnums = torch.tensor(nums).type(torch.LongTensor)
    ptans = torch.tensor(ans).type(torch.FloatTensor)
    
    # specify model and loss
    rnn = RNNModel(rnn_type, len(AB_), ninp, nneur, nlayers, embed = train_embedding)
    
    loss_ = rnn.train(ptnums, ptans, optargs, dlargs, algo=alg, 
                      nepoch=nepoch, criterion=criterion,
                      do_print=False, epsilon=5e-4, padding=pad)
    
    if verbose:
        print('TESTING %s NETWORK' % str(Ls))
    
    
    accuracy = test_net(rnn, list(range(2,11)), AB, forget_switch)
    
    if verbose:
        print('- '*20)
    
    return loss_, accuracy, rnn

def test_net(rnn, these_L, ntest, AB, forget_switch, 
             verbose=False, return_hidden=False):
    
    accuracy = np.zeros(len(these_L))
    if return_hidden:
        H = torch.zeros(2*max(these_L), rnn.nhid, ntest, len(these_L))
        inputs = np.zeros((2*max(these_L), ntest, len(these_L)))
    for j, l in enumerate(these_L):
        if verbose:
            print('On L = %d' %(l))
#        if l in Ls:
#            test_nums = nums[tests,:]
#            test_ans = ans[tests]
#            test_nums = numstest[l]
#            test_ans = anstest[l]
#        else:
        test_nums, test_ans = draw_seqs(l, ntest, Om=AB,
                                        switch=forget_switch,
                                        be_picky=False)
        n_test, lseq = test_nums.shape
        if return_hidden:
            inputs[:lseq,:,j] = test_nums.T
        
        O = torch.zeros(lseq, l, n_test)
        for inp in range(n_test):
            
            tst = test_nums[inp,:]
            
#            if train_embedding:
            enc = tst
            test_inp = torch.tensor(enc).type(torch.LongTensor)
#            else:
#                enc = as_indicator(tst[np.newaxis,:], AB_) #(lenseq, nseq, ninp)
#                test_inp = torch.tensor(enc).type(torch.FloatTensor).transpose(1,0)
            
            hid = rnn.init_hidden(1)
            for t in range(lseq):
                out, hid = rnn(test_inp[t:t+1,...], hid)
                O[t,:,inp] = torch.exp(out[0,0,tst[:l]])
                if return_hidden:
                    H[t,:,inp, j] = hid.squeeze()
                
        O = O.detach().numpy()
        
        whichone = np.argmax(O[-1,:,:],axis=0) # index in sequence of unique number
        accuracy[j] = np.mean(test_nums[np.arange(n_test), whichone] == test_ans.flatten())  

    if return_hidden:
        H = H.detach().numpy()
        return accuracy, H, inputs
    else:
        return accuracy

#%%
def make_dset(Ls, AB, nseq, ntest, forget_switch, padding=0, be_picky=False):
    """
    function to reduce clutter below
    """
    lseq_max = 2*max(Ls)-1
    if forget_switch is not None:
        lseq_max += 1
        
    dset_nums = np.zeros((0,lseq_max))
    dset_nums_test = [[] for _ in range(len(AB))]
    dset_ans = np.zeros(0)
    dset_ans_test = [[] for _ in range(len(AB))]
    
    for l in Ls:
        nums, ans = draw_seqs(l, nseq+ntest, Om=AB, switch=forget_switch,
                              be_picky=be_picky)
        
        trains = np.random.choice(nseq+ntest, nseq, replace=False)
        tests = np.setdiff1d(np.arange(nseq+ntest), trains)
        
        nums_train = nums[trains,:]
        
        foo = np.pad(nums_train, ((0,0),(0,lseq_max-nums.shape[1])), 
                     'constant', constant_values=padding)
        dset_nums = np.append(dset_nums, foo, axis=0)
        dset_ans = np.append(dset_ans, ans[trains])
        
        dset_nums_test[l] = nums[tests,:]
        dset_ans_test[l] = ans[tests]
    
    return dset_nums, dset_ans, dset_nums_test, dset_ans_test

def draw_seqs(L, N, Om=10, switch=-1, be_picky=True, mirror=False):
    """
    Draw N sequences of length 2L-1, such that L-1 elements occur twice
    Optionally specify Om s.t. t_1, ..., t_L \in Om (i.e. |Om| >= L)
    If be_picky=True, it'll return only unique sequences (if few enough)
        - if mirrored, this is at most L*|Om|!/(|Om|-L)!
        - if we allow repeats in any order, it is L!|Om|!/(|Om|-L)!
    
    ToDo: 
        Currently the repetitions are all at the end -- maybe we'll change this
    """
        
    if be_picky:
        if mirror:
            maxseq = int(L*math.factorial(len(Om))/math.factorial(len(Om)-L))
        else:
            maxseq = int(math.factorial(L)*math.factorial(len(Om))/math.factorial(len(Om)-L))
        if N > maxseq: #unlikely
            N = maxseq
        
        # select a random subset of all L-length permutations
        npfx = np.ceil(N/L)
        nperm = math.factorial(len(Om))/math.factorial(len(Om)-L)
        
        perms = permutations(Om, L) # there are many of these
        pinds = np.random.permutation(np.arange(nperm)<npfx)
        toks = np.array(list(compress(perms, pinds))) # select random elements of permutations

        toks = np.repeat(toks, L, axis=0)
        # choose the non-repeated tokens
        inds = (np.tile(np.eye(L), int(npfx)).T > 0)
        
        if mirror: # get repeated tokens
            skot = np.fliplr(toks[inds==0].reshape((-1, L-1))) 
        else:
            skot = scramble(toks[inds==0].reshape((-1, L-1)), axis=0)
        if switch is not None:
            skot = np.insert(skot, 0, switch, axis=1)
            
        S = np.concatenate((toks, skot), axis=1)
        A = toks[inds]
        # scramble
        shf = np.random.choice(int(npfx*L), int(N), replace=False)
        S = S[shf,:]
        A = A[shf]
        
    else: # if there are many unique sequences, just take random samples
        if switch is not None:
            S = np.zeros((N, 2*L), dtype = int)
        else:
            S = np.zeros((N, 2*L-1), dtype = int)
        A = np.zeros((N,1), dtype = int)
        
        # draw tokens from alphabet
        for n in range(N):
            toks = np.random.choice(Om, L, replace = False)
            distok = np.random.choice(L,1)
            skot = np.flip(np.delete(toks,distok))
            if not mirror:
                skot = skot[np.random.choice(L-1,L-1,replace=False)]
            if switch is not None:
                skot = np.append(switch, skot)
            S[n,:] = np.append(toks,skot)
            A[n] = toks[distok]
            
    return S, A

#%% helpers
def as_indicator(x, Om):
    """
    Convert list of sequences x to an indicator (i.e. one-hot) representation
    extra dimensions are added at the end
    """
    def all_idx(idx, axis):
        """ from stack exchange"""
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)
    
    out = np.zeros(x.shape + (len(Om),), dtype = int)
    out[all_idx(x, axis=2)] = 1
    return out

def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    based on unutbu's answer on stack exchange
    """ 
    b = a.swapaxes(axis, -1)
    idx = np.random.rand(*b.shape).argsort(0)
    b = np.take_along_axis(b, idx, 0)
    return b.swapaxes(axis, -1)    

def is_memory_active(seq, mem):
    """
    Returns step function for whether mem is 'active' in seq
    """
    stp = np.cumsum((seq==mem).astype(int)) % 2
    return stp

def unique_seqs(AB, L, mirror = False):
    """
    Return number of unique sequences 
    """
    if mirror:
        maxseq = int(L*math.factorial(len(AB))/math.factorial(len(AB)-L))
    else:
        maxseq = int(math.factorial(L)*math.factorial(len(AB))/math.factorial(len(AB)-L))
    return maxseq

