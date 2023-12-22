import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csc_array
from scipy.sparse import lil_matrix
from scipy.stats import norm

# from functools import partial
import random

#import vectorizers
#import vectorizers.transformers
import pynndescent
# import math

############
# OVERVIEW #
############

# I assume that I start with two objects: "ds_mat" and "data." ds_mat should be a d by s matrix with nonnegative entries, data should be s by n. The motivation is:

# ds_mat: a very sparse matrix with nonnegative entries. I think of this as a "document-sentence" matrix, where ds_mat[i,j] tells me the number of times sentence j appears in document i. 
# Of course this need not be literal: a "document" is any bag of tokens, a "sentence" is any one of those tokens. 
# I use the phrase document-sentence (rather than the more common document-word) to indicate that the tokens are (i) large objects with (ii) almost no overlap. Long n-grams will also "count".

# data: every row should correspond to an embedding of a "sentence" into a common vector space.

# The goal is to end up with a reweighted version of the ds_mat, where sentences are (roughly) ordered by importance. It does so through the steps:
# Make a lower dimensional sentence-kernel matrix
# Turn this into a document-kernel matrix
# Weight the kernels of the document-kernel matrix using IWT
# Lift the weights of the document-kernel matrix to weights of the original document-sentence matrix

###############
# make_sk_mat #
###############

# Data: original sentence-embedding data.
# N: Number of kernels desired.
# exp=25: Each row in data will have roughly exp kernels with nonzero entries.
# metric='euclidean': metric for KNN.
# max_k=40: the maximum value of k that will be used in pynnndescent when computing the KNN index.
# c=1.5: the smallest weight for each kernel will be about e^{-c^{2}}.
# ensure_coverage = False: if true, will run again in batches until every sentence is covered with a kernel.
# batch_size = 10: how large the batch size should be if ensure_coverage=True

# parallel_batch_queries = True: option for pynndescent
# n_trees = 8: option for pynndescent
# max_candidates = 20: option for pynndescent

# Returns:
# sk_mat: the sentence-kernel matrix
# nn_index: the index of all the training data.
# neighbor_query: the query of nn_index for kernel targets. Note that neighbor_query[0][:,0] gives kernel centers
# target_params: the precision value associated with each kernel. 
def make_sk_mat(data,N,exp=25,metric='euclidean',max_k=40,c=1.5,ensure_coverage = False,batch_size = 10, n_trees=8, max_candidates=20,parallel_batch_queries = True):
    # Preliminary Calculations
    n = data.shape[0] # number of sentences
    K = int(exp*(n/N)) # How many nonzero entries each kernel can have

    # Build KNN graph.
    nn_index = pynndescent.NNDescent(pd.DataFrame(data), metric=metric, n_neighbors=min(K,max_k), n_trees=8, max_candidates=20,parallel_batch_queries = True)

    # Find the kernel centers, then the sentences allowed to have nonzero values
    target_inds = [random.randint(0,n-1) for i in range(N)] # The "centers" of the kernels.
    neighbor_query = nn_index.query(data[target_inds,:],k=K)
    nn_inds = neighbor_query[0] # Indices of nearest neighbors
    nn_dists = neighbor_query[1] # Distances to nearest neighbors

    # Find weight parameters, build the sk_mat 
    target_params = nn_dists[:,K-1] # Distance to K'th-nearest-neighbor
    sk_mat = lil_matrix((n,N))
    for i in range(N): # Loop over kernel centers
        p = c/target_params[i]
        for j in range(K): # Loop over sentences allowed to have nonzero values
            sk_mat[nn_inds[i,j],i] = np.exp(-(p*nn_dists[i,j])**2)

    if ensure_coverage: # ZZX: This branch not really tests, as I never use this flag. TODO.
        done = (min(sk_mat.sum(axis=0))==0)
        while not done: # run again
                target_inds = [random.randint(0,n-1) for i in range(batch_size)] # The "centers" of the kernels.
                neighbor_query = nn_index.query(data[target_inds,:],k=K)
                nn_inds = neighbor_query[0] # Indices of nearest neighbors
                nn_dists = neighbor_query[1] # Distances to nearest neighbors
                target_params = nn_dists[:,K-1] # Distance to K'th-nearest-neighbor
                temp_mat = lil_matrix((n,batch_size))
                for i in range(batch_size): # Loop over kernel centers
                    p = c/target_params[i]
                    for j in range(K): # Loop over sentences allowed to have nonzero values
                        temp_mat[nn_inds[i,j],i] = np.exp(-(p*nn_dists[i,j])**2)
                sk_mat = np.vstack(sk_mat,temp_mat)            

    return(sk_mat,nn_index,neighbor_query,target_params)


###############
# make_dk_mat #
###############

# Takes ds_mat, sk_mat and combines them.

def make_dk_mat(ds_mat, sk_mat):
    return(ds_mat@sk_mat)

##############
# Helper: KL #
##############

# A function to calculate the KL divergence between two (unnormalized) distributions.
# Needed for IWT.

def KL(pi,eta, eps = (0.1)**12):
    n = pi.shape[0] # Presumably should check that pi, eta are both nparrays and of the same length
    # normalize
    nPi = np.zeros(n)
    nEta = np.zeros(n)
    zPi = sum(pi) + n*eps
    zEta = sum(eta) + n*eps
    for i in range(n):
        nPi[i] = (pi[i]+eps)/zPi
        nEta[i] = (eta[i] + eps)/zEta
    res = 0
    for i in range(n):
        res += nPi[i]*np.log(nPi[i]/nEta[i])
    return(res)

###############
# Helper: IWT #
###############

# Computes the importance weights for a matrix. Intended to be used for dk_mat, hence the notation.
# NOTE: the Vectorizer library already has such a function, but it has worked quite slowly for many of the (dense) use-cases here. Hence I do a baby re-implementation.
# TO-DO: Reconcile with Vectorizer library.

def IWT(dk_mat, prior_strength=0.1, eps = (0.1)**8):
    D = dk_mat.shape[0] # number of documents
    K = dk_mat.shape[1] # number of kernels

    # Calculate the background or "prior."
    ePrior = dk_mat.sum(axis=1)

    # Calculate the weights
    Weights = np.zeros(K)
    for k in range(K):
        Weights[k] = KL(dk_mat[:,k] + prior_strength*ePrior, ePrior, eps = eps)

    return(Weights)


#################
# weight_lifter #
#################

# sk_mat: nonnegative matrix.
# kern_weights: nonnegative vector, with length equal to the row-length of sk_mat.

# TODO: Come up with a less-silly name. I was thinking Proos.

def weight_lifter(sk_mat, kern_weights, eps = (0.1)**12):
    K = sk_mat.shape[1]
    S = sk_mat.shape[0]
    num = sk_mat@kern_weights
    den = sk_mat.sum(axis=1)
    res = np.zeros(S)
    # TODO: Is this the right way to deal with very small weights? Mostly these are sentences not seen by any kernel, but some might simply be "far."
    for s in range(S):
        if den[s] > eps:
            res[s] = num[s]/den[s]
        else:
            res[s] = 0
    return(res)

#########################
# make_weighted_ds_mat #
#########################

# Makes the weighted ds_mat, using weights from "weightlifter."

def make_weighted_ds_mat(ds_mat,s_weights):
    res = lil_matrix(ds_mat.shape)
    for doc,sent in zip(*ds_mat.nonzero()):
        res[doc,sent] = ds_mat[doc,sent]*s_weights[sent]
    return(res.tocsr())

##############
# do_cts_IWT #
##############

# Obtains weights for a document-sentence matrix.
# Parameters as above
# NOTE ZZX: UNTESTED, but just stringing together the above functions.

def do_cts_IWT(data,ds_mat,N,exp=25,metric='euclidean',max_k=40,c=1.5,ensure_coverage = False,batch_size = 10, n_trees=8, max_candidates=20,parallel_batch_queries = True,eps=(0.1)**12):
    sk_mat,nn_index,neighbor_query,target_params = make_sk_mat(data=data,N=N,exp=exp,metric=metric,max_k=max_k,c=c,ensure_coverage = ensure_coverage,batch_size = batch_size, n_trees=n_trees, max_candidates=max_candidates,parallel_batch_queries = parallel_batch_queries)
    dk_mat = make_dk_mat(ds_mat, sk_mat)
    k_weights = IWT(dk_mat, eps=eps)
    s_weights= weight_lifter(sk_mat, k_weights, eps = eps)
    weighted_ds_mat = make_weighted_ds_mat(ds_mat,s_weights)
    return((s_weights,weighted_ds_mat))


# TO-DO
# Add a little function that takes the "important" sentences to display. Allow it to look at a subset (so we can filter out sentences that are basically all closely associated with a single kernel)
# Add a little function that goes kernel-by-kernel, taking the most representative sentences.
    
    
