import itertools
from itertools import permutations, filterfalse
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.special as spc
import scipy.stats as sts

#%%
def gimme_vertices(v, n):
    """
    Say 'v' is a vertex on an n-cube, represented as a number between 0 and 
    (2^n)-1; (obtained by enumerating the vertices or converting its binary
    representation into decimal).
    
    This function answers the question: what other vertices is it connected to?
    """
    if type(v) is int:
        v = np.array([v])
    elif type(v) is list:
        v = np.array(v)
        
    b = np.array([(v&(2**i))/2**i for i in range(n)]).astype(int).T # convert to binary
    s = 2*(b==0)-1
    p = np.abs(np.sort(np.arange(n)*s, axis=-1))
    
    neigh = v[:,None] + np.sort(s,axis=-1)*(2**p)

    return neigh

def randham(n):
    """
    Draw a Hamiltonian cycle uniformly from an n-cube. **Please, only n<=3.**
    
    A Hamiltonian cycle is a path along a graph which covers all vertices once,
    and there are a heck of a lot of them on the n-cube (http://oeis.org/A066037)
    
    Thanks to (cs.stackexchange.com/users/472/juho) for the algorithm.
    """
    
    def arrbin(v,n):
        """returns n-bit binary representation of v, as an array"""
        return np.array([(v&(2**i))/2**i for i in range(n)]).astype(int).T
     
    isrev = lambda x:(x[0]>x[-1])
    
    i = 1
    allperms = filterfalse(isrev,permutations(np.arange(1,2**n)))
    X = np.zeros((0,(2**n+1)))*np.nan
    while 1:
        try:
            s_ = np.array(next(allperms))
        except StopIteration:
            break
        s = np.concatenate(([0],s_,[0]),axis=0)
        if np.all(la.norm(np.diff(arrbin(s,n),axis=0),1,1)==1): # it is ham
            if np.random.binomial(1,1/i):
                # X = np.append(X,s[None,:],axis=0)
                X = s
                i += 1
    
    return X
    
#%%
P = list(range(2,6))
n_iter=1000

nunq = np.zeros(len(P))
for i,p in enumerate(P):
    all_D = np.zeros((2**(p-1),2,n_iter))
    cube = np.array(list(itertools.product([0,1],repeat=p)))
    for j in range(1000):
        k=0
        while 1: # rejection sampling -- probably not the best...
            g1 = np.sort(np.random.choice(2**p, 2**(p-1), replace=False))
            G1 = cube[g1,:]
            d = la.norm(G1[:,:,None] - G1[:,:,None].transpose(2,1,0), 1, axis=1)
            if ~np.any(np.sum(d==1, axis=1)==p):
                break # no trapped vertices
            else:
                k += 1
        print('partition took %d tries'%k)
        
        g2 = np.setdiff1d(np.arange(2**p), g1)
        G2 = cube[g2,:]
        
        # get all pairs which are only 1 edge away
        d = la.norm(G1[:,:,None] - G2[:,:,None].transpose(2,1,0), 1, axis=1)
        valid = (d==1)
        
        # now, we need to permute such that each pair has distance = 1
        randselect = lambda x:g2[np.random.choice(np.nonzero(x)[0])]
        k = 0
        while 1: # more rejection sampling
            g2_ = np.apply_along_axis(randselect, 1, valid)
            if len(np.unique(g2_)) == len(g2):
                break
            else:
                k += 1
        print('permutation took %d tries'%k)
        
        order = np.argsort(valid.sum(1))
        
        D = np.zeros((0,2),dtype=int)
        for l,g in enumerate(g1[order]):
            g_ = g2[np.random.choice(np.nonzero(valid[order[l],:])[0])]
            
        
#        D = np.zeros((0,2), dtype=int)
#        for g in np.random.permutation(range(2**p)):
#            if g in D:
#                continue
#            
#            available = ~np.isin(np.arange(2**p),D.flatten())
#            
#            g_ = np.random.choice(np.nonzero(valid[g,:]&available)[0],2,replace=False)
#            dg = cube[g_[0],:]-cube[g,:]
#            gg = cube[g_[1],:] + dg
#            _g_ = np.where(la.norm(cube-gg,1,1)==0)[0][0]
#            
#            D = np.append(D, [[g,g_[0]],[g_[1],_g_]], axis=0)
        
        all_D[:,0,j] = g1
        all_D[:,1,j] = g2_
    
    all_D.sort(1)
    nunq[i] = np.unique(all_D.reshape((-1,n_iter)), axis=1).shape[1]

#%% let's plot the adjacency matrix of a p-cube!
p = 12
#col = 'xkcd:orangish red'
#backcol = 'xkcd:pastel purple'
col = 'xkcd:yellow'
backcol = 'xkcd:blood red'

g1 = np.repeat(np.arange(2**p),p)
g2 = gimme_vertices(np.arange(2**p),p).flatten()

s = (np.abs(g2-g1)/np.max(np.abs(g2-g1)))**2.3

plt.scatter(g1,g2, marker='d', s=s, c=col)

plt.gca().set_aspect('equal')
plt.gca().autoscale(tight=True)
plt.gca().invert_yaxis()
plt.gca().set_facecolor(backcol)
plt.gca().autoscale(tight=True)

plt.yticks([])
plt.xticks([])

#%% try manually checking how many dichotomies are 'path partitions'
p = 3

ndic = math.factorial(2**(p-1))*(spc.binom(2**p,2**(p-1))/2)


foo = np.array(list(itertools.combinations(range(2**p),2**(p-1))))

perms = lambda x:list(permutations(x))
G1 = np.repeat(foo[:int(foo.shape[0]/2),:],math.factorial(2**(p-1)),axis=0)
G2 = np.flipud(np.apply_along_axis(perms,1,foo[int(foo.shape[0]/2):,:]).reshape((-1,4)))

# check whether each pair in G1 and G2 are connected by an edge
bin1 = np.array([(G1&(2**i))/2**i for i in range(p)]).astype(int) # convert
bin2 = np.array([(G2&(2**i))/2**i for i in range(p)]).astype(int) # to binary
# count them
d = la.norm(bin1-bin2, 1, axis=0)
npart = np.sum(np.all(d==1,axis=1))

#%% try randomly drawing Hamiltonian cycles

def hipow(n):
    ''' Return the highest power of 2 within n. '''
    exp = 0
    while 2**exp <= n:
        exp += 1
    return 2**(exp-1)

def code(n):
    ''' Return nth gray code. '''
    if n>0:
        return hipow(n) + code(2*hipow(n) - n - 1)
    return 0

# main:
for n in range(30):
    print(bin(code(n))[2:])
    
    
    
    
    
    
    
    