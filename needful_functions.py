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
from students import RNNModel

#%%
def train_and_test(rnn_type, N, P, Ls, nseq, ntest, explicit=True,
                   nlayers=1, pad=-1, dlargs=None, optargs=None,
                   criterion=None, alg=None, be_picky=False, nepoch=2000,
                   train_embedding=True, verbose=True):
    """
    function to reduce clutter below
    """
    
    AB = list(range(P))
    nneur = N                  # size of recurrent network
    if explicit:
        forget_switch =  max(AB)+1
    else:
        forget_switch = None
    
    ninp = len(AB)              # dimension of RNN input
    
    if type(Ls) is not list:
        Ls = [Ls]
    
    # optimisation parameters
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
                                             ntest=int(ntest/len(Ls)), 
                                             forget_switch=forget_switch,
                                             padding=-1,
                                             be_picky=be_picky)
    nums[nums==-1] = pad
    
    ptnums = torch.tensor(nums).type(torch.LongTensor)
    ptans = torch.tensor(ans).type(torch.FloatTensor)
    
    # specify model and loss
    rnn = RNNModel(rnn_type, len(AB_), ninp, nneur, nlayers, embed = train_embedding)
    
    loss_ = rnn.train(ptnums, ptans, optargs, dlargs, algo=alg, 
                      nepoch=nepoch, criterion=criterion,
                      do_print=False, epsilon=5e-4, padding=pad)
    
    if verbose:
        print('TESTING %s NETWORK' % str(Ls))
    
    if be_picky:
        accuracy = test_net(rnn, AB, test_data=(numstest, anstest), 
                            forget_switch=forget_switch)
    else:
        accuracy = test_net(rnn, AB, Ls, ntest, forget_switch=forget_switch)
    
    if verbose:
        print('- '*20)
    
    return loss_, accuracy, rnn

def test_net(rnn, AB, these_L=None, ntest=None, test_data=None, padding=-1,
             explicit=True, forget_switch=None, verbose=False, 
             return_hidden=False, give_gates=False):
    """
    Run `rnn` on `ntest` sequences of L in `these_L`, drawn from alphabet `AB` 
    Returns the accuracy on the test sequences, and optionally the hidden 
    activity+input sequence and/or gate activations. 
    
    If `ntest` isn't specified, you can instead give `test_data`, which is a 
    tuple of (seqs, ans).
    
    ONLY SET give_gates=True IF RNN IS OF CLASS tanh-GRU!! 
    
    Outputs in order: (accuracy, hidden, inputs, (update, reset))
    """
    
    if explicit and forget_switch is None:
        forget_switch = max(AB)+1
    elif not explicit:
        forget_switch = None
    
#    lseqmax = 2*max(these_L)-1*(not explicit)
    
#    accuracy = np.zeros(len(these_L))
#    if return_hidden:
#        H = torch.zeros(lseqmax, rnn.nhid, ntest, len(these_L))
#        inputs = np.zeros((lseqmax, ntest, len(these_L)))
#    if give_gates:
#        Z = torch.zeros(lseqmax, rnn.nhid, ntest, len(these_L))
#        R = torch.zeros(lseqmax, rnn.nhid, ntest, len(these_L))
#    
    if ntest is not None:
        test_nums, test_ans, _, _ = make_dset(these_L, AB, ntest, 0, 
                                              forget_switch=forget_switch,
                                              be_picky=False, 
                                              padding=padding)
    else:
        test_nums, test_ans = test_data
        ntest = test_nums.shape[0]
        
    ignore = test_nums==padding
    test_inp = torch.tensor(test_nums).type(torch.LongTensor)
    test_inp[ignore] = 0
    test_inp = test_inp.transpose(0,1)
    
    lseq = (~ignore).sum(1)
    t_final = lseq-1 # which is the final timestep
    
    # actually test
    hid = rnn.init_hidden(test_inp.shape[1])
            
    if give_gates:
        out, H, ZR = rnn(test_inp, hid, give_gates=True)
        Z, R = ZR
        O = torch.exp(out)
    else:
        out, H = rnn(test_inp, hid)
        O = torch.exp(out)
    
#    for j, l in enumerate(these_L):
#        if verbose:
#            print('On L = %d' %(l))
#        test_nums, test_ans = draw_seqs(l, ntest, Om=AB,
#                                        switch=forget_switch,
#                                        be_picky=False)
#        n_test, lseq = test_nums.shape
#        if return_hidden:
#            inputs[:lseq,:,j] = test_nums.T
        
#        O = torch.zeros(lseq, l, n_test)
#        for inp in range(n_test):
#            
#            tst = test_nums[inp,:]
#            
#            enc = tst
#            test_inp = torch.tensor(enc).type(torch.LongTensor)
#            
#            hid = rnn.init_hidden(1)
#            for t in range(lseq):
#                if give_gates:
#                    out, hid, ZR = rnn(test_inp[t:t+1,...], hid, give_gates=True)
#                    Z[t,:,inp,j] = ZR[0].squeeze()
#                    R[t,:,inp,j] = ZR[1].squeeze()
#                else:
#                    out, hid = rnn(test_inp[t:t+1,...], hid)
#                O[t,:,inp] = torch.exp(out[0,0,tst[:l]])
#                if return_hidden:
#                    H[t,:,inp, j] = hid.squeeze()
    
    O = O.detach().numpy()
    O_T = O[t_final,np.arange(ntest),:] # output at final timestep
    whichone = O_T.argmax(1) # network's answer

    accuracy = [np.mean(whichone[lseq==l] == test_ans[lseq==l]) for l in np.unique(lseq)]
    
#    accuracy = np.mean(test_nums[np.arange(ntest), whichone] == test_ans.flatten())
#    whichone = np.argmax(O[-1,:,:],axis=0) # index in sequence of unique number
#    accuracy = np.mean(test_nums[np.arange(n_test), whichone] == test_ans.flatten())  
    
    outputs = (accuracy, )
    if return_hidden:
        H = H.detach().numpy()
        outputs += (H, test_nums)
    if give_gates:
        outputs += (Z.detach().numpy(), R.detach().numpy())
    
    return outputs

#%%
def make_dset(Ls, AB, nseq, ntest, forget_switch=None, padding=0, be_picky=False):
    """
    Wrapper for draw_seqs, which concatenates sequences of multiple lengths into
    a single dataset. Fills extra space with `padding`, which you should keep
    track of when you feed this data into an RNN.
    
    If `be_picky=True`, then train and test sets are disjoint -- but heed the 
    warning in the `draw_seqs` help text.
    
    returns seqs, ans, test_seqs, test_ans
    """
    lseq_max = 2*max(Ls)-1
    if forget_switch is not None:
        lseq_max += 1
        
    dset_nums = np.zeros((0,lseq_max))
    dset_nums_test = np.zeros((0,lseq_max))
    dset_ans = np.zeros(0)
    dset_ans_test = np.zeros(0)
    
    for l in Ls:
        nums, ans = draw_seqs(l, nseq+ntest, Om=AB, switch=forget_switch,
                              be_picky=be_picky)
#        print('drawn')
        if nums.shape[0] < nseq+ntest:
            Nseq = int(nseq*nums.shape[0]/(nseq+ntest))
            Ntest = nums.shape[0]-Nseq
        else:
            Nseq = nseq
            Ntest = ntest
        
        trains = np.random.choice(Nseq+Ntest, Nseq, replace=False)
        tests = np.setdiff1d(np.arange(Nseq+Ntest), trains)
    
        train_seqs = np.pad(nums[trains,:], ((0,0),(0,lseq_max-nums.shape[1])),
                            'constant', constant_values=padding)
        test_seqs = np.pad(nums[tests,:], ((0,0),(0,lseq_max-nums.shape[1])),
                           'constant', constant_values=padding)
#        print('padded')
        dset_nums = np.append(dset_nums, train_seqs, axis=0)
        dset_ans = np.append(dset_ans, ans[trains])
        
        dset_nums_test = np.append(dset_nums_test, test_seqs, axis=0)
        dset_ans_test = np.append(dset_ans_test, ans[tests])
    
    return dset_nums, dset_ans, dset_nums_test, dset_ans_test

def draw_seqs(L, N, Om, switch=-1, be_picky=False, mirror=False):
    """
    Draw N sequences of length 2L, such that L-1 elements occur twice, and
    all elements are drawn from Om.
    
    If be_picky=True, it'll return only unique sequences (if few enough)
        - if mirrored, this is at most L*|Om|!/(|Om|-L)!
        - if we allow repeats in any order, it is L!|Om|!/(|Om|-L)!
        - please please please consider how many sequences you're generating
            * at some point, we need to index the set of all possible 
              permutations of Om of length L, so that gets big quickly
            * roughly: don't be picky if |Om|!/(|Om|-L)! > ~10^7
    
    ToDo: 
        Currently the repetitions are all at the end -- maybe we'll change this
    """
        
    if be_picky:
        maxseq = unique_seqs(len(Om), L, mirror)
        if N > maxseq: # if P and L are sufficiently small
            N = maxseq
            
        # select a random subset of all permutations of length L (or all of them)
        # (we'll take different strategies for subsampling and exhaustive sampling)
        nperm = math.factorial(len(Om))/math.factorial(len(Om)-L) # number of permutations
        if N == maxseq:
            npfx = nperm
        else:
            npfx = np.ceil(N/L) # number of 'prefixes'
        
        perms = permutations(Om, L) # there are many of these
        pinds = np.random.permutation(np.arange(nperm)<npfx) # shuffled binary indexing
        toks = np.array(list(compress(perms, pinds))) # select random elements of permutations
        
        toks = np.repeat(toks, L, axis=0)

        # choose the non-repeated tokens
        inds = (np.tile(np.eye(L), int(min([nperm,npfx]))).T > 0)

        A = toks[inds]
        skot = toks[inds==0].reshape((-1, L-1))
        if npfx >= nperm: # i.e. we need more 
            # add more sequences by permuting the forget phase
            nsubperm = math.factorial(L-1)
            d = int(np.ceil(N/(L*nperm))) # how many permutations to use (always <= (L-1)!)
            
            toks = np.repeat(toks, d, axis=0)
            A = np.repeat(A, math.factorial(L-1), axis=0)
            
            sigma = np.random.permutation # for convenience
            allperms = lambda x:list(compress(permutations(x, L-1),\
                                              sigma(np.arange(nsubperm)<d)))
            
            skot = np.apply_along_axis(allperms, 1, skot).reshape((-1, L-1))
        else:
            if mirror: # get repeated tokens
                skot = np.fliplr(skot) 
            else:
                skot = scramble(skot, axis=0)
                
        if switch is not None:
            skot = np.insert(skot, 0, switch, axis=1)
            
        S = np.concatenate((toks, skot), axis=1)
        # scramble
        shf = np.random.choice(S.shape[0], int(N), replace=False)
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
            toks = np.random.choice(Om, L, replace=False)
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

def unique_seqs(P, L, mirror = False):
    """
    Return number of unique sequences 
    """
    if mirror:
        maxseq = int(L*math.factorial(P)/math.factorial(P-L))
    else:
        maxseq = int(math.factorial(L)*math.factorial(P)/math.factorial(P-L))
    return maxseq

