CODE_DIR = r'/home/matteo/Documents/github/rememberforget/'
svdir = r'~/Documents/uni/columbia/sueyeon/experiments/'

import sys, os
sys.path.append(CODE_DIR)

from sklearn import svm, calibration, linear_model, discriminant_analysis
import scipy.stats as sts
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.linalg as la
import umap
from cycler import cycler

from students import RNNModel, stateDecoder

from needful_functions import *

#%% Load from Habanero experiment
#fracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,3.0]
fracs = [1,2,3,4,5,6,7,8,9,10,15,20,30]
Ps = [10]
L_ = 5
rnn_type = 'GRU'
explicit = True
smoothening = 12
embed = False

folds = ''
if explicit:
    folds += 'explicit/'
else:
    folds += 'implicit/'
if embed:
    folds += 'embedded/'

accuracy = np.zeros((len(Ps),len(fracs),4))*np.nan
#Ls = np.zeros((len(Ps),9))
loss = np.zeros((len(Ps),len(fracs),200000))

rnns = [[] for _ in range(len(fracs))]
for i,P in enumerate(Ps):
#    L = int(L_*P/10)
#    nnnn = int(np.ceil((P-3)/8))
#    test_L = list(range(2,P))
#    test_L.insert(4,L)
#    test_L = [L-2,L,L+2]
#    test_L += list(range(L+3, P, int(np.ceil((P-L)/2))))
#    test_L = list(range(2, L-2, int(np.ceil((P-L)/2)))) + test_L
#    test_L.append(P-1)
    test_L = [2,3,4,5]
#    Ls[i,:] = np.array(test_L)
    
    for j,N in enumerate(fracs):
#        N = int(f*P)
        
        params_fname = 'parameters_%d_%d_%s.pt'%(N, P, rnn_type)
        loss_fname = 'loss_%d_%d_%s.npy'%(N, P, rnn_type)
        if not os.path.exists(CODE_DIR+'results/'+folds+loss_fname):
            print(CODE_DIR+'results/'+folds+loss_fname+' doesn`t exist!')
            continue
        
        rnn = RNNModel(rnn_type, P+1*explicit, P+1*explicit, N, 1, embed=embed)
        rnn.load(CODE_DIR+'results/'+folds+params_fname)
        rnns[j] = rnn
#        test_L = [l for l in range(L-3,L+4)]
#        test_L.append(P)
#        test_L.insert(0,2)
#        test_L = list(range(2,P))
            
#        results = test_net(rnn, list(range(P)), test_L, 500, explicit=explicit)[0]
        
        print('done testing N=%d, P=%d network'%(N,P))
        
        los = np.load(CODE_DIR+'results/'+folds+loss_fname)
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
plt.legend(np.array(fracs), loc='upper right')
plt.ylabel('NLL loss')
plt.title('Training with P='+str(Ps[pid]))

#%%
pid = 0
plt.figure()
plt.plot(test_L, accuracy[pid,:,:].T)
plt.legend(fracs)
plt.plot(test_L, 1/np.array(test_L), 'k--')
plt.xlabel('test L')
plt.ylabel('test accuracy')

#%% Linear decoding of memory activation
L = [3]
whichrnn = -1
P = Ps[pid]

isgru = rnn_type in ['GRU', 'tanh-GRU']
if isgru:
    rnn = RNNModel('tanh-GRU', P+1*explicit, P+1*explicit, fracs[whichrnn], 1, embed=embed)
    rnn.load(CODE_DIR+'results/'+folds+'parameters_%d_%d_GRU.pt'%(fracs[whichrnn], P))
else:
    rnn = rnns[whichrnn]

nseq = 1500
#clsfr = calibration.CalibratedClassifierCV(svm.LinearSVC(tol=1e-5, max_iter=5000), cv=10)
#clsfr = svm.LinearSVC
#cfargs = {'tol': 1e-5, 'max_iter':5000}
#clsfr = linear_model.LogisticRegression(tol=1e-5, solver='lbfgs', max_iter=1000)
clsfr = discriminant_analysis.LinearDiscriminantAnalysis
cfargs = {}

# generate training data
acc, H, I = test_net(rnn, L, nseq, list(range(P)), 
                     explicit=explicit, return_hidden=True)
lseq = H.shape[0]

#H = H.transpose(1,0,2,3).reshape((H.shape[1],-1))
H = H.squeeze() # lenseq x nneur x nseq
I = I.squeeze() # lenseq x nseq
H_flat = H.transpose(1,0,2).reshape((H.shape[1],-1))

ans = np.zeros(I.shape + (P+1,)) # lenseq x nseq x ntoken
for p in range(P+1):
    ans[:,:,p] =np.apply_along_axis(is_memory_active, 0, I, p)

# do the decoding
clf = [[clsfr(**cfargs) for _ in range(lseq)] for _ in range(P+1)]

for t in range(lseq):
    for p in range(P):
        clf[p][t].fit(H[t,...].T, ans[t,:,p])
    if explicit:
        clf[-1][t].fit(H_flat.T, ans[:,:,-1].flatten())
    else:
        tim = np.tile(np.arange(lseq)[:,np.newaxis],(1,H.shape[2]))
        clf[-1][t].fit(H_flat.T, tim.flatten()>=L[0])

# generate test data
if isgru:
    acc, H, I, Z, R = test_net(rnn, L, nseq, list(range(P)), 
                               explicit=explicit,
                               return_hidden=True, give_gates=True)
    Z = Z.squeeze() # lenseq x nneur x nseq
    R = R.squeeze() # lenseq x nneur x nseq
else: 
    acc, H, I = test_net(rnn, L, nseq, AB=list(range(P)), 
                         explicit=explicit, return_hidden=True)

H = H.squeeze() # lenseq x nneur x nseq
I = I.squeeze() # lenseq x nseq

ans = np.zeros(I.shape + (P+1,)) # lenseq x nseq x ntoken
for p in range(P+1):
    ans[:,:,p] =np.apply_along_axis(is_memory_active, 0, I, p)

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

#%% see how well the LDs separate token activations
plot_these = list(range(lseq))
#plot_these = [1, L[0], -1]
#p_fixed = 1

fig, axs = plt.subplots(P, len(plot_these))
for p in range(P):
    axs[p][0].set_ylabel('Token %d'%p)
    for t,tt in enumerate(plot_these):
        if p==0:
            axs[p,t].set_title('Time %d'%tt)
            
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
plot_these = list(range(lseq))
#plot_these = [1, L[0], -1]

fig, axs = plt.subplots(1, len(plot_these))
for t,tt in enumerate(plot_these):
    axs[t].imshow(inner[:,:,tt], cmap='bwr')
    axs[t].set_title('Time %d'%tt)
    axs[t].set_xlabel('Token')
    axs[t].get_images()[0].set_clim([-1,1])
    if t != 0:
        axs[t].set_yticks([])

axs[0].set_ylabel('Token')
#fig.colorbar()

fig, axs = plt.subplots(1, len(plot_these))
for t,tt in enumerate(plot_these):
    axs[t].imshow(coefs[:,:,tt].T, cmap='bwr')
    axs[t].set_title('Time %d'%tt)
    axs[t].set_xlabel('Token')
    axs[t].get_images()[0].set_clim([-1,1])
    if t != 0:
        axs[t].set_yticks([])

axs[0].set_ylabel('Neuron')

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

if isgru:
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.imshow(Z[:,:,which_seq].T, cmap='binary')
    ax1.get_images()[0].set_clim([0,1])
    ax1.set_title('Update gate activities')
    
    ax2.imshow(R[:,:,which_seq].T, cmap='binary')
    ax2.get_images()[0].set_clim([0,1])
    ax2.set_title('Reset gate activities')
    plt.colorbar(ax2.get_images()[0])

#%%
#foo = la.norm(coefs - coefs[:,:,:1], axis=1)
foo = la.norm(np.diff(coefs, axis=-1), axis=1)
plt.plot(list(range(1,lseq)),foo.T)
plt.legend(list(range(P+1)))
plt.ylabel('Distance from previous timestep')
plt.xlabel('time')

plt.figure()
plt.plot(range(lseq), thrs.T)
plt.legend(list(range(P+1)))
plt.ylabel('Classifier decision boundary')
plt.xlabel('time')

plt.figure()
zproj = np.einsum('ikj...,jk...->ij...', coefs, Z) # projection onto LDs


#%% Decode time in trial
L = [7]
rnn = rnns[-3]
P = Ps[pid]

nseq = 1500
#clsfr = calibration.CalibratedClassifierCV(svm.LinearSVC(tol=1e-5, max_iter=5000), cv=10)
clsfr = svm.LinearSVC
#clsfr = linear_model.LogisticRegression(tol=1e-5, solver='lbfgs', max_iter=1000)
#clsfr = discriminant_analysis.LinearDiscriminantAnalysis
cfargs = {'tol': 1e-5, 'max_iter':5000}

# generate training data
acc, H, I = test_net(rnn, L, nseq, list(range(P)), 
                     explicit=explicit, return_hidden=True)
lseq = H.shape[0]

#H = H.transpose(1,0,2,3).reshape((H.shape[1],-1))
H = H.squeeze() # lenseq x nneur x nseq
I = I.squeeze() # lenseq x nseq
H_flat = H.transpose(1,0,2).reshape((H.shape[1],-1))

ans = np.tile(np.arange(lseq)[:,np.newaxis],(1,H.shape[2]))
ans_flat = ans.flatten()

# do the decoding
clf = clsfr(**cfargs)
clf.fit(H_flat.T, ans_flat)

# test
acc, H, I = test_net(rnn, L, nseq, list(range(P)), 
                     explicit=explicit, return_hidden=True)
H = H.squeeze() # lenseq x nneur x nseq
I = I.squeeze() # lenseq x nseq
H_flat = H.transpose(1,0,2).reshape((H.shape[1],-1))

plt.figure()
plt.plot(clf.predict(H_flat.T))
plt.plot(ans_flat, 'k--')
plt.legend(['predicted time','actual time'])
plt.xlabel('time point, ordered by position in trial')
plt.ylabel('position in trial')
plt.title('Accuracy: %.3f'%clf.score(H_flat.T, ans_flat))

#%%
C = clf.coef_/la.norm(clf.coef_, axis=-1)[:, np.newaxis]
inner = C@C.T
proj = C@H_flat
intcpt = clf.intercept_/la.norm(clf.coef_, axis=-1)

plt.figure()
plt.imshow(C.T, cmap='bwr')
plt.gca().get_images()[0].set_clim([-1,1])
plt.xlabel('time point')
plt.ylabel('neuron')
plt.title('Classifier weights')
plt.colorbar()

# center the projections around the decision boundary
fig, axs = plt.subplots(2,int(lseq/2))
for t in range(lseq):
    r = round((t+1)/lseq)
    c = t%int(lseq/2)
    axs[r,c].hist(proj[t, ans_flat==t], density=True, alpha=0.5, color='r')
    for tt in range(lseq):
        if tt != t:
            axs[r,c].hist(proj[t, ans_flat == tt], density=True, alpha=0.5)
    axs[r,c].set_title('Time %d'%t)
    axs[r,c].legend(np.concatenate(([t], np.setdiff1d(range(lseq),t))))
    axs[r,c].plot([intcpt[t], intcpt[t]], axs[r,c].get_ylim(), 'k-')
    if c!=0:
        axs[r,c].set_yticks([])
    if r!=1:
        axs[r,c].set_xticks([])
    axs[r,c].set_xlabel('Proj onto t=%d  weights'%t)


## <whinging>
## I'm a real big fan of how there is a set of methods in mpl.pyplot, and the 
## TOTALLY EQUIVALENT methods in mpl.axes.Axes mostly have DIFFERENT NAMES
## it makes my life SO MUCH EASIER
##
## And a side-note on python syntax,
## To concatenate vector 'a' to matrix 'B' in:
## numpy: np.concatenate((a[np.newaxis, :], B))
## matlab: [a, B] 
## </whinging>

#%% Looking at projections of the activity
#X = H.transpose(1,0,2).reshape((H.shape[1],-1)) # hidden
X = Z.transpose(1,0,2).reshape((Z.shape[1],-1)) # update gate
#X = R.transpose(1,0,2).reshape((R.shape[1],-1)) # reset gate

tim = np.tile(np.arange(lseq)[:,np.newaxis],(1,H.shape[2]))
tim = tim.flatten()
trial = np.tile(np.arange(nseq)[:,np.newaxis],(1,H.shape[0]))
trial = trial.T.flatten() 
token = I.flatten()
firsthalf = tim <= L[0]

# do UMAP
reducer = umap.UMAP(n_components=2)
emb = reducer.fit_transform(X.T)

# do PCA
_, S, V = la.svd(X.T-X.mean(1), full_matrices=False)
pcs = X.T@V[:3,:].T

#%% PCA
cmap_name = 'nipy_spectral'
colorby = token
#colorby = tim
#colorby = firsthalf

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter(pcs[:,0],pcs[:,1],pcs[:,2], c=colorby, alpha=0.1, cmap=cmap_name)
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('pc3')

cb = plt.colorbar(scat, 
                  ticks=np.unique(colorby),
                  drawedges=True,
                  values=np.unique(colorby))
cb.set_ticklabels(np.unique(colorby))
cb.set_alpha(1)
cb.draw_all()

#plt.figure()
#plt.

#%% UMAP
cmap_name = 'nipy_spectral'
#colorby = token
#colorby = tim
colorby = firsthalf

# plot
scat = plt.scatter(emb[:,0],emb[:,1], c=colorby, alpha=0.1, cmap=cmap_name)

cb = plt.colorbar(scat, 
                  ticks=np.unique(colorby),
                  drawedges=True,
                  values=np.unique(colorby))
cb.set_ticklabels(np.unique(colorby))
cb.set_alpha(1)
cb.draw_all()
#cb.set_label('Time in trial')

plt.xlabel('umap dimension 1')
plt.ylabel('umap dimension 2')

#%%
tok = 8
tr = 13

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(emb[:,0],emb[:,1], c=[[0.5,0.5,0.5]], alpha=0.01)
ax.scatter(emb[trial==tr, 0],emb[trial==tr, 1], c='b')

for t, i in enumerate(zip(I[:,0], emb[trial==tr, :])):
    ax.annotate('t=%d: %d'%(t,i[0]), i[1])

ax.legend(['not  trial', 'trial'])

plt.xlabel('umap dimension 1')
plt.ylabel('umap dimension 2')

#ax.scatter(emb[firsthalf & istok_on,0], 
#           emb[firsthalf & istok_on,1], 
#           emb[firsthalf & istok_on,2])
#ax.scatter(emb[firsthalf & istok_off,0], 
#           emb[firsthalf & istok_off,1], 
#           emb[firsthalf & istok_off,2])
#
#ax.scatter(emb[~firsthalf & istok_on,0], 
#           emb[~firsthalf & istok_on,1],
#           emb[~firsthalf & istok_on,2])
#ax.scatter(emb[~firsthalf & istok_off,0], 
#           emb[~firsthalf & istok_off,1],
#           emb[~firsthalf & istok_off,2])

#plt.scatter(emb[firsthalf & istok_on,0],emb[firsthalf & istok_on,1])
#plt.scatter(emb[firsthalf & istok_off,0],emb[firsthalf & istok_off,1])
#plt.scatter(emb[~firsthalf & istok_on,0],emb[~firsthalf & istok_on,1])
#plt.scatter(emb[~firsthalf & istok_off,0],emb[~firsthalf & istok_off,1])

#ax.scatter(emb[:,0],emb[:,1], c=[[0.5,0.5,0.5]], alpha=0.01)




