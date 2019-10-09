CODE_DIR = r'/home/matteo/Documents/github/rememberforget/'
svdir = r'~/Documents/uni/columbia/sueyeon/experiments/'

import sys
sys.path.append(CODE_DIR)

from sklearn import svm, calibration, linear_model, discriminant_analysis
import scipy.stats as sts
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.linalg as la
from cycler import cycler

from students import RNNModel, stateDecoder
from needful_functions import *

#%% Load from Habanero experiment
fracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,3.0]
Ps = [10]
L_ = 5
rnn_type = 'GRU'
smoothening = 12

accuracy = np.zeros((len(Ps),len(fracs),24))*np.nan
Ls = np.zeros((len(Ps),8))
loss = np.zeros((len(Ps),len(fracs),200000))

rnns = [[] for _ in range(len(fracs))]
for i,P in enumerate(Ps):
#    L = int(L_*P/10)
#    nnnn = int(np.ceil((P-3)/8))
#    test_L = list(range(3,P,2))
#    test_L = [L-2,L,L+2]
#    test_L += list(range(L+3, P, int(np.ceil((P-L)/2))))
#    test_L = list(range(2, L-2, int(np.ceil((P-L)/2)))) + test_L
#    test_L.append(P-1)
#    Ls[i,:] = np.array(test_L)
    
    for j,f in enumerate(fracs):
        N = int(f*P)
        
        params_fname = 'parameters_%d_%d_%s.pt'%(N, P, rnn_type)
        loss_fname = 'loss_%d_%d_%s.npy'%(N, P,  rnn_type)
        
        rnn = RNNModel('GRU', P+1, P+1, N, 1, embed=True)
        rnn.load(CODE_DIR+'results/'+params_fname)
        rnns[j] = rnn
#        test_L = [l for l in range(L-3,L+4)]
#        test_L.append(P)
#        test_L.insert(0,2)
#        test_L = list(range(2,P))
        
#        results = test_net(rnn, test_L, 500, list(range(P)), P)
        
        print('done testing N=%d, P=%d network'%(N,P))
        
        los = np.load(CODE_DIR+'results/'+loss_fname)
        idx = np.min([200000, los.shape[0]])
        
        loss[i,j,:idx] = los[:idx]
#        accuracy[i,j,:] = results

loss_smooth = np.apply_along_axis(np.convolve,-1,loss,np.ones(smoothening),'same')

#%% 
pid = 0
cmap_name = 'ocean'
cols = getattr(cm, cmap_name)(np.linspace(0,1,len(fracs)))
mpl.rcParams['axes.prop_cycle'] = cycler(color=cols)
plt.plot(loss_smooth[pid,:,:].T)
#plt.plot()
plt.legend(np.array(fracs)*Ps[pid], loc='upper right')
plt.ylabel('NLL loss')
plt.title('Training with P='+str(Ps[pid]))

#%% Linear decoding of memory activation
L = [5]
rnn = rnns[-3]
P = Ps[pid]

nseq = 1500
#clsfr = calibration.CalibratedClassifierCV(svm.LinearSVC(tol=1e-5, max_iter=5000), cv=10)
#clsfr = linear_model.LogisticRegression(tol=1e-5, solver='lbfgs', max_iter=1000)
clsfr = discriminant_analysis.LinearDiscriminantAnalysis

# generate training data
acc, H, I = test_net(rnn, L, nseq, list(range(P)), P, return_hidden=True)
lseq = H.shape[0]

#H = H.transpose(1,0,2,3).reshape((H.shape[1],-1))
H = H.squeeze() # lenseq x nneur x nseq
I = I.squeeze() # lenseq x nseq
H_flat = H.transpose(1,0,2).reshape((H.shape[1],-1))

ans = np.zeros(I.shape + (P+1,)) # lenseq x nseq x ntoken
for p in range(P+1):
    ans[:,:,p] =np.apply_along_axis(is_memory_active, 0, I, p)
#ans = ans.transpose(1,0,2).reshape((-1,P+1))

# do the decoding
clf = [[clsfr() for _ in range(lseq)] for _ in range(P+1)]

token_probs = np.zeros(ans.shape)
for t in range(lseq):
    for p in range(P):
        clf[p][t].fit(H[t,...].T, ans[t,:,p])
        token_probs[t,:,p] = clf[p][t].predict_proba(H[t,...].T)[:,1]
    clf[-1][t].fit(H_flat.T, ans[:,:,-1].flatten())

# generate test data
acc, H, I = test_net(rnn, L, nseq, list(range(P)), P, return_hidden=True)

#H = H.transpose(1,0,2,3).reshape((H.shape[1],-1))
H = H.squeeze() # lenseq x nneur x nseq
I = I.squeeze() # lenseq x nseq

ans = np.zeros(I.shape + (P+1,)) # lenseq x nseq x ntoken
for p in range(P+1):
    ans[:,:,p] =np.apply_along_axis(is_memory_active, 0, I, p)
#ans = ans.transpose(1,0,2).reshape((-1,P+1))

# test performance
perf = np.array([[clf[p][t].score(H[t,:,:].T, ans[t,:,p]) for t in range(lseq)] for p in range(P)])

plt.imshow(perf)
plt.colorbar()

plt.title('decoder accuracy (test sequences, per-time decoders)')
plt.xlabel('time')
plt.ylabel('token')

#%% code for SVM
mpl.rcParams['axes.prop_cycle'] = cycler(color='rgbcymk')
which_seq = 2

these_seqs = np.isin(list(range(P+1)), I[:,which_seq])   
plt.plot(token_probs[:,which_seq,these_seqs])
plt.xticks(ticks=range(lseq),labels=I[:,which_seq])
plt.plot([L,L],plt.ylim(), 'k-')

plt.legend(np.arange(P+1)[these_seqs])
plt.title('Memory activations in example sequence')
plt.xlabel('Token presented at time t')
plt.ylabel('Probability of being present')

plt.figure()
plt.imshow(H[:,:,which_seq].T)
#plt.plot(H[:,:,which_seq])
plt.xticks(ticks=range(lseq),labels=I[:,which_seq])
plt.plot([L,L],plt.ylim(), 'k-')
plt.title('Neural activations in example sequence')
plt.xlabel('Token presented at time t')
plt.ylabel('Neuron')

#%% Code for LDA
# get the projections onto LDs, and angle between different LDs
coefs = np.zeros((P+1, H.shape[1], lseq))
thrs = np.zeros((P+1,lseq))
for p in range(P+1):
    for t in range(lseq):
        coefs[p,:,t] = clf[p][t].coef_/la.norm(clf[p][t].coef_)
        thrs[p,t] = -clf[p][t].intercept_/la.norm(clf[p][t].coef_)

inner = np.einsum('ik...,jk...->ij...', coefs, coefs) # dot product between LDs
proj = np.einsum('ikj...,jk...->ij...', coefs, H) # projection onto LDs

# center the projections around the decision boundary
proj_ctr = proj - thrs[:,:,np.newaxis]

#%% see how well the LDs separate memories
plot_these = list(range(lseq))
#plot_these = [1, L[0], -1]
#p_fixed = 1

fig, axs = plt.subplots(P, len(plot_these))
for p in range(P):
    axs[p][0].set_ylabel('Token %d'%p)
    for t,tt in enumerate(plot_these):
        if p==0:
            axs[p,t].set_title('Time %d'%tt)
#            
#        C = clf[p][-2].coef_/la.norm(clf[p][-2].coef_)
#        ison = C.dot(H[tt,...])[:,ans[tt,:,p]==1].squeeze()
#        isoff = C.dot(H[tt,...])[:,ans[tt,:,p]==0].squeeze()
        ison = proj_ctr[p,tt,ans[tt,:,p]==1]
        isoff = proj_ctr[p,tt,ans[tt,:,p]==0]
        foo = 0
#        pval = sts.ks_2samp(ison,isoff).pvalue
#        foo = -clf[p][1].intercept_/la.norm(clf[p][1].coef_)
        
        axs[p,t].hist(ison, density=True, alpha=0.5)
        axs[p,t].hist(isoff, density=True, alpha=0.5)
        axs[p,t].plot([foo,foo],axs[p,t].get_ylim(), 'k-')
#        axs[p,t].text(axs[p,t].get_xlim()[0],axs[p,t].get_ylim()[1],
#                 'p={:.1e} (K-S)'.format(pval), fontsize=10, va='top')
        
        if p==P-1:
            axs[p][t].set_xlabel('LD1')

#%% see how different LDs relate to each other
fig, axs = plt.subplots(1, len(plot_these))
for t,tt in enumerate(plot_these):
    axs[t].imshow(inner[:,:,tt], cmap='bwr')
    axs[t].set_title('Time %d'%tt)
    axs[t].set_xlabel('Token')
    axs[t].get_images()[0].set_clim([-1,1])

axs[0].set_ylabel('Token')
#fig.colorbar()

#%% how do the projections evolve for example sequences?
mpl.rcParams['axes.prop_cycle'] = cycler(color='rgbcymk')
which_seq = 40

these_toks = np.isin(list(range(P+1)), I[:,which_seq]) 

plt.plot(proj_ctr[these_toks,:,which_seq].T)
plt.xticks(ticks=range(lseq),labels=I[:,which_seq])
plt.plot([L,L],plt.ylim(), 'k-')
plt.plot(plt.xlim(),[0,0], 'k--')
plt.legend(np.arange(P+1)[these_toks])
plt.title('Projections onto best LD of each token & time')
plt.ylabel('Projection (centred at decision boundary)')
#plt.ylabel('Projection')
plt.xlabel('Token at time t')

#%%
#foo = la.norm(coefs - coefs[:,:,:1], axis=1)
foo = la.norm(np.diff(coefs, axis=-1), axis=1)
plt.plot(list(range(1,lseq)),foo.T)
plt.legend(list(range(P+1)))
plt.ylabel('Distance from previous timestep')
plt.xlabel('time')



