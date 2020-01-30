"""
Here go all the functions that I will want in the simulations and subsequent 
analysis of the remember-forget project. 

Maybe this is a remnant of my MatLab days, but there are just so many little 
functions I want when working with these simulations, and I can't be bothered 
to make a class or something.
"""

import sys, os, re
#sys.path.append(CODE_DIR)

import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import compress, permutations
#from joblib import Parallel, delayed
#import multiprocessing
import numpy as np
import scipy.special as spc
from students import RNNModel, Indicator

#%%
def train_and_test(rnn_type, N, P, Ls, nseq, ntest, nlayers=1, pad=-1, 
                   expargs=None, dlargs=None, optargs=None,
                   criterion=None, alg=None, nepoch=2000, dead_time=None,
                   train_embedding=False, persistent=False, 
                   return_data=False, verbose=True):
    """
    function to reduce clutter 
    """
    
    AB = list(range(P))
    nneur = N             # size of recurrent network
    ninp = P              # dimension of RNN input
    
    if type(Ls) is not list:
        Ls = [Ls]
    
    ## DEFAULTS --------------------------------
    # sequence parameters
    if expargs is None: # parameters for `draw_seqs`
        expargs = {} # will propagate function defaults
        expargs['explicit'] = True
        expargs['be_picky'] = False
        expargs['forget'] = True
        if verbose:
            print('Using default sequence parameters'
                  'be_picky: false, context_cue: explicit')

    explicit = expargs['explicit']
    be_picky = expargs['be_picky']
    
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
    # -------------------------------------------
    
    if explicit or persistent:
        ninp += 1
    
    # check this before proceding
    if be_picky and (math.factorial(max(Ls))/math.factorial(P-max(Ls)))>1e7:
        be_picky = False
        expargs['be_picky'] = None
        print("Hey! What are you doing!? That's too many sequences, my friend.")
        print("Setting to pickiness default")
    
    # train and test
    if verbose:
        print('TRAINING %s NETWORK' % str(Ls))
    
    nums, ans, numstest, anstest = make_dset(Ls, AB, nseq, 
                                             ntest=ntest,
                                             padding=pad,
                                             dead_time=dead_time,
                                             **expargs)
    
    # expargs['be_picky']=None
    # _, _, numstest_, anstest_ = make_dset([5,20], AB, nseq, 
    #                                        ntest=ntest,
    #                                        padding=pad,
    #                                        dead_time=dead_time,
    #                                        **expargs)
#    nums[nums==-1] = pad
    
    ptnums = torch.tensor(nums).type(torch.LongTensor)
    ptans = torch.tensor(ans).type(torch.FloatTensor)
    
    test_nums = torch.tensor(numstest,requires_grad=False).type(torch.LongTensor)
    test_ans = torch.tensor(anstest,requires_grad=False).type(torch.FloatTensor)
    
    # specify model and loss
    rnn = RNNModel(rnn_type, ninp, ninp, nneur, nlayers, 
                   embed=train_embedding, persistent=persistent,
                   padding=pad)
    
    rnn.train(ptnums, ptans, optargs, dlargs, algo=alg, 
              nepoch=nepoch, criterion=criterion,
              do_print=False, epsilon=5e-4,
              test_data=(test_nums,test_ans))
    
#    loss_ = rnn.train_loss
#    test_loss_ = rnn.test_loss
    
    if verbose:
        print('TESTING %s NETWORK' % str(Ls))
        
    if be_picky: # ensure that test set is different
        accuracy = test_net(rnn, AB, test_data=(numstest, anstest),
                            dead_time=dead_time, **expargs)[0]
    else:
        accuracy = test_net(rnn, AB, Ls, ntest, dead_time=dead_time, **expargs)[0]
    
    if verbose:
        print('- '*20)
    
    if return_data:
        return accuracy, rnn, (nums, ans, numstest, anstest)
    else:
        return accuracy, rnn

def test_net(rnn, AB, these_L=None, ntest=None, test_data=None, padding=-1,
             verbose=False, return_hidden=False, give_gates=False, dead_time=None,
             **expargs):
    """
    Run `rnn` on `ntest` sequences of L in `these_L`, drawn from alphabet `AB` 
    Returns the accuracy on the test sequences, and optionally the hidden 
    activity+input sequence and/or gate activations. 
    
    If `ntest` isn't specified, you can instead give `test_data`, which is a 
    tuple of (seqs, ans).
    
    ONLY SET give_gates=True IF RNN IS OF CLASS tanh-GRU!! 
    
    Outputs in order: (accuracy, hidden, inputs, (update, reset))
    """
    
#    if (ntest is None) and (test_data is None):
#        raise(ValueError)
    
    if ntest is not None:
        test_nums, test_ans, _, _ = make_dset(these_L, AB, ntest, 0,
                                              padding=rnn.padding,
                                              dead_time=dead_time,
                                              **expargs)
    else:
        test_nums, test_ans = test_data
        
    Ntest = test_nums.shape[0]

    ignore = torch.tensor(test_nums==rnn.padding)
    test_inp = torch.tensor(test_nums).type(torch.LongTensor)
#    print(test_inp.shape)
#    print(ignore.shape)
#    test_inp[ignore] = 0
    test_inp = test_inp.transpose(0,1) # lseq x ntest
    lseq = (~ignore).sum(1)
    t_final = -(np.fliplr(test_nums!=rnn.padding).argmax(1)+1) # which is the final timestep
    lmax = test_nums.shape[1]
    
    # test predictions
    if return_hidden:
        H = torch.zeros(test_nums.shape[1], rnn.nhid, Ntest)
    if give_gates:
        Z = torch.zeros(lmax, rnn.nhid, Ntest)
        R = torch.zeros(lmax, rnn.nhid, Ntest)
    
    hid = rnn.init_hidden(test_inp.shape[1])
    if return_hidden or give_gates:
        # because pytorch only returns hidden activity in the last time step,
        # we need to unroll it manually. 
        O = torch.zeros(lmax, Ntest, rnn.decoder.out_features)
        emb = rnn.encoder(test_inp)
        for t in range(lmax):
            if give_gates:
                out, hid, ZR = rnn.rnn(emb[t:t+1,...], hid, give_gates=True)
                Z[t,:,:] = ZR[0].squeeze(0).T
                R[t,:,:] = ZR[1].squeeze(0).T
            else:
                out, hid = rnn.rnn(emb[t:t+1,...], hid)
            dec = rnn.decoder(out)
            H[t,:,:] = hid.squeeze(0).T
            O[t,:,:] = dec.squeeze(0)
    else: # this is faster somehow
        O, _ = rnn(test_inp, hid)
    
    O = O.detach().numpy()
    O_T = O[t_final,np.arange(Ntest),:] # output at final timestep, NseqxNout
    
    if test_ans.ndim > 1:
        acc = nn.BCEWithLogitsLoss(reduction='none')(torch.tensor(O_T).double(), 
                                   torch.tensor(test_ans).double()).detach().numpy()
#        outpt = O_T.argsort(1)
        accuracy = [acc[lseq==l,:].mean(0) for l in np.unique(lseq)]
    else:
        whichone = O_T.argmax(1) # network's answer
        accuracy = [np.mean(whichone[lseq==l] == test_ans[lseq==l]) for l in np.unique(lseq)]
    
    # collect for output
    outputs = (accuracy, )
    if return_hidden:
        H = H.detach().numpy()
        outputs += (H, test_nums.T)
    if give_gates:
        outputs += (Z.detach().numpy(), R.detach().numpy())
    
    return outputs

def batch_test(this_folder, rnn_type, P, L, N_list=None, Q_list=None):
    """
    Load all the parameters for RNNs saved in this_folder, and test the networks.
    
    ONLY WORKS FOR THE SINGLE-CONTEXT EXPERIMENTS
    """
    
    if (N_list is None) or (Q_list is None):
        files = os.listdir(this_folder)
        param_files = [f for f in files if 'parameters' in f]
        
        if len(param_files)==0:
            raise ValueError('No RNNS in specified folder `^`')
        
        rex = r"parameters_(\d+)_(\d+)_?(\d+(?:-\d)*)?"
        NP = np.array([re.findall(rex,f)[0][:2] for f in param_files]).astype(int)
        Ls = [(re.findall(rex,f)[0][2] or '0') for f in param_files]
        q = np.array([re.findall(r"Q(\d+)",f) or ['0'] \
                      for f in param_files]).astype(int).squeeze()
#        if L is not None:
        isL = np.isin(Ls, L) # only take experiments with the right L
#        isQ = np.isin(q, Q)
#        else:
#            isL = np.ones(NP.shape[0])>1
        if N_list is None:
            N_list = np.unique(NP[isL&(NP[:,1]==P),0])
#        if P_list is None:
#            P_list = np.unique(NP[isL&(q==Q),1])
        has_n = np.isin(NP[:,0],N_list)
        if Q_list is None:
            Q_list = np.unique(q[isL&(NP[:,1]==P)&has_n])
    
    L = list(np.array(L.split('-')).astype(int))
#    signal_tokens=None
    
    accuracy = np.zeros((len(Q_list),len(N_list), P, P))*np.nan
    trained_Ls = [[] for _ in Q_list]
    metrics = {}
    # metrics = {'train_loss':np.zeros((len(Q_list),len(N_list),150000))*np.nan,
    #            'test_loss':np.zeros((len(Q_list),len(N_list),150000))*np.nan,
    #            'train_orthog':np.zeros((len(Q_list),len(N_list),150000))*np.nan,
    #            'test_orthog':np.zeros((len(Q_list),len(N_list),150000))*np.nan,
    #            'test_parallelism':np.zeros((len(Q_list),len(N_list),150000))*np.nan}
    
    rnns = [[[] for _ in N_list] for _ in Q_list]
    for i,Q in enumerate(Q_list):
        
        for j,N in enumerate(N_list):
            
            expinf = suffix_maker(N, P, L, rnn_type, Q, None)
            params_fname = 'parameters'+expinf+'.pt'
            loss_fname = 'loss'+expinf+'.npy'
            metrics_fname = 'metrics'+expinf+'.pkl'
            args_fname = 'arguments'+expinf+'.npy'
            accy_fname = 'accuracy'+expinf+'.npy'
            dset_fname = 'datasets'+expinf+'.pkl'
            
            if not os.path.exists(this_folder+params_fname):
                print(this_folder+params_fname+' doesn`t exist!')
                continue
            
            ninp = P
            rnn = RNNModel(rnn_type, P, P, N, 1, embed=False, 
                           persistent=False)
            rnn.load(this_folder+params_fname)
            rnns[i][j] = rnn
            test_L = list(range(2,P))
            if os.path.exists(this_folder+args_fname):
                dsargs = np.load(this_folder+args_fname, allow_pickle=True).item()
                trained_Ls[i] = dsargs['Ls']
                if type(dsargs['Ls']) is not list:
                    dsargs['Ls'] = [dsargs['Ls']]
                these_L = [l for l in test_L if l not in dsargs['Ls']] # setdiff
        #           these_L = test_L
                if len(these_L) != 0:
                    acc1 = test_net(rnn, list(range(P)), these_L, 500, 
                                    be_picky=None, signal_tokens=None,
                                    explicit=False, forget=False)[0]
                    accuracy[i,j,these_L,:P] = acc1
                    
                acc2 = np.load(this_folder+accy_fname)
                accuracy[i,j,dsargs['Ls'],:P] = acc2
                
            else:
                acc1 = test_net(rnn, list(range(P)), test_L, 500,
                                be_picky=None, signal_tokens=None,
                                explicit=False, forget=False)[0]
                accuracy[i,j,test_L,:P] = acc1
            
    #        Ls[test_L]
            
            print('done testing N=%d, Q=%d network'%(N,Q))
            
            if os.path.exists(this_folder+metrics_fname):
                met = pickle.load(open(this_folder+metrics_fname,'rb'))
                for key, val in met.items():
                    if key not in metrics.keys():
                        if 'parallelism' in key:
                            metrics[key] = np.zeros((max(Q_list),len(Q_list),len(N_list),150000))*np.nan
                        else:
                            metrics[key] = np.zeros((len(Q_list),len(N_list),150000))*np.nan
            else:
                los = np.load(this_folder+loss_fname)
                if os.path.exists(this_folder+'test_'+loss_fname):
                    tlos = np.load(this_folder+'test_'+loss_fname)
                    
            if os.path.exists(this_folder+metrics_fname):
                for key, val in met.items():
                    idx = np.min([150000, val.shape[0]])
                    # metrics[key][i,j,:idx] = val.reshape((val.shape[0],-1))[:idx].mean(1)
                    if 'parallelism' in key:
                        jdx = val.shape[1]
                        metrics[key][:jdx,i,j,:idx] = val[:,:idx].T
                    else:
                        metrics[key][i,j,:idx] = val[:idx]
                # metrics['test_loss'][i,j,:idx] = tlos[:idx]
                # metrics['test_parallelism'][i,j,:idx] = pllm[:idx]
                # metrics['train_orthog'][i,j,:idx] = orth[:idx]
                # metrics['test_orthog'][i,j,:idx] = torth[:idx]
            elif os.path.exists(this_folder+'test_'+loss_fname):
                metrics['test_loss'][i,j,:idx] = tlos[:idx]
    
    metrics['train_loss'] = np.apply_along_axis(
            np.convolve,-1,metrics['train_loss'],np.ones(10),'same')
    metrics['test_loss'] = np.apply_along_axis(
            np.convolve,-1,metrics['test_loss'],np.ones(10),'same')
    
    return (N_list, Q_list, trained_Ls), (rnns, metrics, accuracy)

#%%
def make_dset(Ls, AB, nseq, ntest, padding=-1, dead_time=None, **expargs):
    """
    Wrapper for draw_seqs, which concatenates sequences of multiple lengths into
    a single dataset. Fills extra space with `padding`, which you should keep
    track of when you feed this data into an RNN. Specifically, the value of 
    `padding` should be the same as `padding` in the RNNModel.train method.
    (The defaults in both functions are matched, so don't worry about it.)
    
    If `be_picky=True`, then train and test sets are disjoint -- but heed the 
    warning in the `draw_seqs` help text.
    
    `dead_time` specifies the total amount of time spent after tokens are
    presented -- i.e. how much time to distribute across 
    
    returns seqs, ans, test_seqs, test_ans
    """
    
    if 'forget' in expargs.keys(): # not ideal way to do this
        f = expargs['forget']
        lseq_max = 2*max(Ls)
    else:
        f = None
        lseq_max = max(Ls)
    
    ntot = nseq+ntest
    if 'be_picky' in expargs.keys():
        if expargs['be_picky']:
            # for some L,P combinations, there will be very few unique sequences
            # we need to ensure that the total # of unique seqs is still nseq+ntest
            nseqs = np.array([np.min([unique_seqs(len(AB),l,forget=f), ntot]) for l in Ls])/ntot
            nrich = max([1,np.sum(nseqs[nseqs==1])])
            nseqs[nseqs==1] = (1-nseqs[nseqs<1].sum())/nrich
            nseqs = np.round(nseqs*ntot).astype(int) # maybe there's a better way to divvy...
        else:
            nseqs = np.round(ntot*np.ones(len(Ls))/len(Ls)).astype(int)
    else:
         nseqs = np.round(ntot*np.ones(len(Ls))/len(Ls)).astype(int)
    
#    dset_nums = []
#    dset_nums_test = []
#    dset_ans = []
#    dset_ans_test = []
    dset_nums = np.zeros((0,lseq_max))
    dset_nums_test = np.zeros((0,lseq_max))
    dset_ans = np.zeros((0,len(AB))) # will need to make this compatible 
    dset_ans_test = np.zeros((0,len(AB))) # with the forget sequences
    
    for i,l in enumerate(Ls):
        Nseq = int(round(nseqs[i]*(nseq/ntot)))
        Ntest = nseqs[i] - Nseq
#        print((Nseq, Ntest))
        nums, ans = draw_seqs(l, nseqs[i], Om=AB, **expargs)
#        print('drawn')
#        if nums.shape[0] < nseq+ntest:
#            Nseq = int(nseq*nums.shape[0]/(nseq+ntest))
#            Ntest = nums.shape[0]-Nseq
#        else:
#            Nseq = nseq
#            Ntest = ntest
        
        trains = np.random.choice(Nseq+Ntest, Nseq, replace=False)
        tests = np.setdiff1d(np.arange(Nseq+Ntest), trains)
        train_seqs = np.pad(nums[trains,:], ((0,0),(0,lseq_max-nums.shape[1])),
                            'constant', constant_values=padding)
        test_seqs = np.pad(nums[tests,:], ((0,0),(0,lseq_max-nums.shape[1])),
                           'constant', constant_values=padding)
#        print('padded')
#        if Ls.index(l) == 0:
#            dset_nums = train_seqs
#            dset_ans = ans[trains]
#            dset_nums_test = test_seqs
#            dset_ans_test = ans[tests]
#        else: 
        dset_nums = np.append(dset_nums, train_seqs, axis=0)
        dset_ans = np.append(dset_ans, ans[trains], axis=0)
        dset_nums_test = np.append(dset_nums_test, test_seqs, axis=0)
        dset_ans_test = np.append(dset_ans_test, ans[tests], axis=0)
        
#        dset_nums += train_seqs.tolist()
#        dset_ans += ans[trains].tolist()
#        
#        dset_nums_test += test_seqs.tolist()
#        dset_ans_test += ans[tests].tolist()
    
    dset_nums = dset_nums[:, ~np.all(dset_nums==padding, axis=0)]
    dset_nums_test = dset_nums_test[:, ~np.all(dset_nums_test==padding, axis=0)]
#    dset_ans = dset_ans.squeeze()
#    dset_ans_test = dset_ans_test.squeeze()
    
    if dead_time is not None:
        nT = dset_nums.shape[1] + dead_time
        
        _dset_nums = np.ones((dset_nums.shape[0],nT))*padding
        idx = np.random.rand(dset_nums.shape[0],nT).argsort(1)[:,:(nT-dead_time)]
        idx = np.sort(idx,1)
        idx[:,0] = 0
#        print(idx)
        np.put_along_axis(_dset_nums, idx, dset_nums, axis=1)
        dset_nums = _dset_nums
        
        if dset_nums_test.shape[0] > 0:
            _dset_nums_test = np.ones((dset_nums_test.shape[0],nT))*padding
            idx = np.random.rand(dset_nums_test.shape[0],nT).argsort(1)[:,:(nT-dead_time)]
            idx = np.sort(idx,1)
            idx[:,0] = 0
            np.put_along_axis(_dset_nums_test, idx, dset_nums_test, axis=1)
            dset_nums_test = _dset_nums_test
    
    return dset_nums, dset_ans, dset_nums_test, dset_ans_test

def draw_seqs(L, N, Om, explicit=True, be_picky=None,
              signal_tokens=None, forget=True):
    """
    Draw N sequences of length 2L, such that L-1 elements occur twice, and
    all elements are drawn from Om.
    
    If be_picky=True, it'll return only unique sequences (if few enough)
        - if mirrored, this is at most L*|Om|!/(|Om|-L)!
        - if we allow repeats in any order, it is L!|Om|!/(|Om|-L)!
        - if nothing is supplied (None), it will choose based on P and L
        - please please please consider how many sequences you're generating
            * at some point, we need to index the set of all possible 
              permutations of Om of length L, so that gets big quickly
            * roughly: don't be_picky if |Om|!/(|Om|-L)! > ~10^7
    
    Supplying `signal_tokens` will further restrict the sequences to those with
    non-repeated tokens that are members of signal_tokens. For example, if 
    signal_tokens = [0,1] then only 0 or 1 will be non-repeated. Right now, 
    this only works for be_picky=True, and takes substantially longer. In any
    case, this will severely restrict the number of unique sequences available.
    
    TODO: 
        Currently the repetitions are all at the end -- maybe we'll change this
    """
    if signal_tokens is None:
        signal_tokens = Om
        
    if explicit:
        forget_switch = len(Om)
    else:
        forget_switch= None
    
    P = len(Om)
    X = len(signal_tokens)
    
    if be_picky is None:
        try:
            be_picky = math.factorial(P)/math.factorial(P-L) < 1e7
        except OverflowError:
            be_picky = False
            
#    print(X)
    if be_picky:
        maxseq = unique_seqs(P, L, X, forget=forget)
        if N > maxseq: # if P and L are sufficiently small
            N = maxseq
        
        # select a random subset of all permutations of length L (or all of them)
        # (we'll take different strategies for subsampling and exhaustive sampling)
        nperm_tot = math.factorial(P)/math.factorial(P-L)
        nperm = round(pintersection(P,L,X)*nperm_tot) # number of permutations
        if forget:
            if N == maxseq:
                npfx = nperm
            else:
                if X != P:
                    # get minimum possible number of 
                    nx = min([n for n in range(1,X+1) if pintersection(P,L,X,n)>0])
                    npfx = np.ceil(N/nx)
                else:
                    npfx = np.ceil(N/L) # number of 'prefixes'        
        else:
            npfx = N
        
        if X != P:
            checkifin = lambda x:np.any(np.isin(x, range(X)))
            perms = filter(checkifin,permutations(Om, L)) # there are many of these
        else:
            perms = permutations(Om, L)
        # select a random subset of permutations
        pinds = np.random.permutation(np.arange(nperm)<npfx) # shuffled binary indexing
        toks = np.array(list(compress(perms, pinds))) # select random elements of permutations
        
        # choose the non-repeated tokens
        inds = (np.tile(np.eye(L), int(min([nperm,npfx]))).T > 0)

        if X != P:
            sig_toks = np.isin(toks, range(X)) # ensure we only select 'signal' tokens
            inds = inds[sig_toks.flatten(),:] # as the non-repeated ones
        
        if not forget:
            A = Indicator(P,P)(toks).sum(1).detach().numpy()
            return toks, A
        
        if X != P:
            toks = np.repeat(toks, sig_toks.sum(1), axis=0)
        else:
            toks = np.repeat(toks, L, axis=0)
        
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
            skot = scramble(skot, axis=0)
        
        if forget_switch is not None:
            skot = np.insert(skot, 0, forget_switch, axis=1)

        S = np.concatenate((toks, skot), axis=1)

        # scramble
        shf = np.random.choice(S.shape[0], int(N), replace=False)
        S = S[shf,:]
        A = A[shf]
        
    else: # if there are many unique sequences, just take random samples
        toks = np.random.rand(N,P).argsort(1)[:,:L] # generate prefixes
        
        if not forget:
            A = Indicator(P,P)(toks).sum(1).detach().numpy()
            return toks, A
            
        skot = scramble(toks[:,:-1], axis=0) # generate suffices
        if forget_switch is not None:
            skot = np.insert(skot, 0, forget_switch, axis=1)
            
        A = toks[:,-1]
        toks = scramble(toks, axis=0)
        
        S = np.concatenate((toks, skot), axis=1)
        
        shf = np.random.choice(S.shape[0], int(N), replace=False)
        S = S[shf,:]
        A = A[shf]
        
    return S, A

#%%
def draw_hidden_for_decoding(rnn, L, nseq, chunking=None,
                             decode_token=False, decode_from='hidden', 
                             **seq_args):
    """
    Take hidden activity from rnn in response to drawn sequences.     
    """
    
    P = rnn.decoder.out_features
    
    if 'GRU' in str(type(rnn.rnn)):
        acc, H, I, Z, R = test_net(rnn, list(range(P)), L, nseq,
                                   return_hidden=True, give_gates=True, 
                                   **seq_args)
        H = H.squeeze() # lenseq x nneur x nseq
        I = I.squeeze() # lenseq x nseq
        Z = Z.squeeze() # lenseq x nneur x nseq
        R = R.squeeze() # lenseq x nneur x nseq
#        hidden = (H,I,Z,R)
    else: 
        acc, H, I = test_net(rnn, list(range(P)), L, nseq, return_hidden=True,
                             **seq_args)
        H = H.squeeze() # lenseq x nneur x nseq
        I = I.squeeze() # lenseq x nseq
#        hidden = (H,I)
        
    lseq = H.shape[0]
    nseq = H.shape[2]
    
    these_t = (I!=rnn.padding)
    
    mem_act = np.zeros(I.shape + (P,)) # lenseq x nseq x ntoken
    tok_pres = np.zeros(I.shape + (P,)) # lenseq x nseq x ntoken
    
    idx = np.nonzero((I.T!=-1)*(I.T!=P))[1].reshape((nseq,-1))
    nrep = np.diff(idx, axis=1, append=lseq)
    tok = np.take_along_axis(I,np.repeat(idx.flatten(),nrep.flatten()).reshape((nseq,-1)).T,0)
    for p in range(P):
        mem_act[:,:,p] = np.apply_along_axis(is_memory_active, 0, I, p)
        tok_pres[:,:,p] = (tok==p)
    
    if chunking == 'timepoints':
        t_ = (np.cumsum(np.ones((lseq, nseq)), axis=0)-1).astype(int)
    elif chunking == 'tokens':
        t_ = np.cumsum(I!=rnn.padding, axis=0)-1
    elif chunking == 'context':
        t_ = np.cumsum(I!=rnn.padding, axis=0)-1
        t_[t_<L] = 0
        t_[t_>=L] = 1
    elif (chunking == 'none') or (chunking is None):
        t_ = np.zeros((lseq, nseq), dtype=int)
    
    idx = np.nonzero(these_t.T)[1].reshape((nseq,-1)).T
    if decode_from == 'hidden':
        X = np.take_along_axis(H, idx[:,None,:], 0)
    elif decode_from == 'reset':
        X = np.take_along_axis(R, idx[:,None,:], 0)
    elif decode_from == 'update':
        X = np.take_along_axis(Z, idx[:,None,:], 0)
    t_ = np.take_along_axis(t_, idx, 0)
    
    if decode_token:
        ans = np.take_along_axis(tok_pres, idx[:,:,None], 0)
    else:
        ans = np.take_along_axis(mem_act, idx[:,:,None], 0)
    
    
    return X, I, t_, ans, acc

def parallelism(H, C):
    """
    Computes the parallelism score from the Geometry of Abstraction paper
    (Bernardini, ..., Fusi 2019)
    
    H is the hidden activity (T x N) and C is (T x M), where M is the
    number of dichotomies being considered (i.e. number of coding vectors); it
    should be binary, indicating which time belongs to which class.
    """
    
    m = C.shape[1]
    
    V = np.array([H[C[:,c],:].mean(0)-H[~C[:,c],:].mean(0) for c in range(m)])
    V = V/la.norm(V,axis=1)[:,None]
    csin = V.dot(V.T)
    PS = np.triu(csin,1).mean()
    
    return PS

#%% file I/O
def expfolders(exp, pers, emb, forg=True):
    """Folder hierarchy for organising results
        - expfolders(explicit?, persistent?, embed?)"""
    folds = 'results/'
    if not forg:
        folds += 'justremember/'
    else:
        if exp:
            folds += 'explicit/'
        else:
            folds += 'implicit/'
        if pers:
            folds += 'persistent/'
    if emb:
        folds += 'embedded/'
    return folds

def suffix_maker(N, P, L, rnn_type, Q=None, dead_time=None):
    """
    A consistent naming scheme to make it, supposedly, easier to find outputs
    of experiments.
    
    Will give something like 'X_Y_Z' to be added before the file extension.
    e.g., I might do 
        `filename = 'accuracy' + suffix_maker(10, 10, 'GRU') + '.npy'
        print(filename)`
    which would output
        `accuracy_10_10_GRU.npy`
        
    Can add further things whenever you want, but for compatibility, try to 
    keep the `_N_P_rnn` kernel intact -- unless it demonstrably sucks.
    """
    suff = '_%d_%d_'%(N, P)
    suff += ((len(L)*'%d-')%tuple(L))[:-1]
    suff += '_%s'%rnn_type
    if Q is not None:
        suff += '_Q%d'%Q
#    if L is not None:
#        suff += '_L%d'%L
    if dead_time is not None:
        suff += '_dt%d'%dead_time
    
    return suff

##%% plotting
#def myplot(cmap=None, cols==None, **mplargs):
#    
#    
#    plt.plot(**mplargs)

#%% helpers
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

def was_token_presented(seq, mem):
    """
    Returns step function for whether mem was the last token presented
    """
    idx = np.nonzero(seq!=-1)[0]
    stp = seq[np.repeat(idx,np.diff(idx,append=len(seq)))] == mem
    return stp

def unique_seqs(P, L, X=None, n=None, forget=True):
    """
    Return number of unique remember(/forget) sequences 
    """
    if X is None:
        X = P
    if not forget:
        maxseq = int(pintersection(P,L,X,n)*(math.factorial(P)/math.factorial(P-L)))
    else:
#        if X is not None:
    #            maxseq = math.factorial(P)/math.factorial(P-L)
    #            maxseq *= sum([n*pintersection(P,L,X,n) for n in range(1,X+1)])
        maxseq = math.factorial(L)*math.factorial(P)/math.factorial(P-L)
        maxseq = int(maxseq*pintersection(P,L,X,n))
#        else:
#            maxseq = int(math.factorial(L)*math.factorial(P)/math.factorial(P-L))
    return maxseq

def pintersection(P,L,X,n=None):
    """
    Say there is a set with P elements, and a subset of it with X < P elements;
    this computes the probability that another subset with L elements shares n 
    elements with the first subset. If n is unspecified, returns prob(n > 0).
    """
    if n is None:
        return 1-spc.binom(P-X,L)/spc.binom(P,L)
    else:
        return spc.binom(P,n)*spc.binom(P-X,L-n)*spc.binom(P-n,X-n)/(spc.binom(P,L)*spc.binom(P,X))


