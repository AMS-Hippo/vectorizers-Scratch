import math
import numpy as np
import pandas as pd
from functools import partial
import pynndescent
import random
import scipy.sparse
from scipy.sparse import csc_array
from scipy.sparse import lil_matrix
from scipy.stats import norm

###############
# make_sk_mat #
###############

# X: original sentence-embedding data.
# N: Number of kernels desired.
# exp=25: The expected number of nonzero entries in each row of the returned matrix is roughly exp.
# metric='euclidean': metric for KNN.
# max_k=40: the maximum value of k that will be used in pynnndescent when computing the KNN index.
# c=1.5: the smallest weight for each kernel will be about e^{-c^{2}}.
# ensure_coverage = False: if true, will run again in batches until every sentence is covered with a kernel.
# batch_size = 10: how large the batch size should be if ensure_coverage=True
# pow = 2: decay rate for kernel. By default, pow=2 gives Gaussian kernels.

# parallel_batch_queries = True: option for pynndescent
# n_trees = 8: option for pynndescent
# max_candidates = 20: option for pynndescent

# Returns:
# sk_mat: the sentence-kernel matrix
# nn_index: the index of all the training .
# neighbor_query: the query of nn_index for kernel targets. Note that neighbor_query[0][:,0] gives kernel centers
# target_params: the precision value associated with each kernel. 

def make_sk_mat(data,N=None,exp=25,metric='euclidean',max_k=40,c=1.5,ensure_coverage = False,batch_size = 10, n_trees=8, max_candidates=20,parallel_batch_queries = True,pow=2):
    # Preliminary Calculations
    n = data.shape[0] 
    if N is None:
        N = min(n,math.ceil(math.log(n,2)*800)) # Default option inspired by JL lemma with error epsilon =0.1, which is fairly arbitrary. TODO: Experiment.
    K = int(exp*(n/N)) # How many nonzero entries each kernel can have


    # Build KNN graph.
    nn_index = pynndescent.NNDescent(pd.DataFrame(data), metric=metric, n_neighbors=min(K,max_k), n_trees=n_trees, max_candidates=max_candidates,parallel_batch_queries = parallel_batch_queries)

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

    if ensure_coverage: # TODO: This branch hasn't been tested much.
        done = (min(sk_mat.sum(axis=0))==0)
        while not done: # run again
                target_inds = [random.randint(0,n-1) for i in range(batch_size)] # The "centers" of the kernels.
                neighbor_query = nn_index.query(X[target_inds,:],k=K)
                nn_inds = neighbor_query[0] # Indices of nearest neighbors
                nn_dists = neighbor_query[1] # Distances to nearest neighbors
                target_params = nn_dists[:,K-1] # Distance to K'th-nearest-neighbor
                temp_mat = lil_matrix((n,batch_size))
                for i in range(batch_size): # Loop over kernel centers
                    p = c/target_params[i]
                    for j in range(K): # Loop over sentences allowed to have nonzero values
                        temp_mat[nn_inds[i,j],i] = np.exp(-(p*nn_dists[i,j])**pow)
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

######################
# Helper: Simple_IWT #
######################

# Computes the importance weights for a matrix. Intended to be used for dk_mat, hence the notation.
# This is a simpler version of the main IWT method. TODO: At some point I expect this to be phased out.

def Simple_IWT(dk_mat, prior_strength=0.1, eps = (0.1)**8):
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

#######################
# get_cts_IWT_weights #
#######################

# Obtains weights for a document-sentence matrix and the final weighted matrix.
# Parameters as above
# TODO: UNTESTED, but it is "just" stringing together the above functions.

def get_cts_IWT_weights(data,ds_mat,N=None,exp=25,metric='euclidean',max_k=40,c=1.5,ensure_coverage = False,batch_size = 10, n_trees=8, max_candidates=20,parallel_batch_queries = True,eps=(0.1)**12,simple_IWT=False):
    sk_mat,nn_index,neighbor_query,target_params = make_sk_mat(data=data,N=N,exp=exp,metric=metric,max_k=max_k,c=c,ensure_coverage = ensure_coverage,batch_size = batch_size, n_trees=n_trees, max_candidates=max_candidates,parallel_batch_queries = parallel_batch_queries)
    dk_mat = make_dk_mat(ds_mat, sk_mat)
    if simple_IWT:
        k_weights = Simple_IWT(dk_mat, eps=eps) # TODO: Use the actual IWT from the library, not this stub method.
    else:
        k_weights = Simple_IWT(dk_mat, eps=eps)
    s_weights= weight_lifter(sk_mat, k_weights, eps = eps)

    return((k_weights,s_weights))

######################################
# Helper: read_local_gaussian_params #
######################################

# Expands the parameters when using read_local_gaussian_params

def read_local_gaussian_params(method_params):
    if method_params is None:
        N = None
        exp=30
        metric='euclidean'
        max_k=40
        c=1.5
        ensure_coverage = False
        batch_size = 10
        n_trees=8
        max_candidates=20
        parallel_batch_queries = True
        eps=(0.1)**12
        pow = 2
    else:
        if 'N' in method_params.keys():
            N=method_params['N']
        else:
            N = None
        if 'exp' in method_params.keys():
            exp=method_params['exp']
        else:
            exp = 30        
        if 'metric' in method_params.keys():
            metric=method_params['metric']
        else:
            metric = 'euclidean'
        if 'max_k' in method_params.keys():
            max_k=method_params['max_k']
        else:
            max_k=40
        if 'c' in method_params.keys():
            c=method_params['c']
        else:
            c=1.5
        if 'ensure_coverage' in method_params.keys():
            ensure_coverage=method_params['ensure_coverage']
        else:
            ensure_coverage = False
        if 'batch_size' in method_params.keys():
            batch_size=method_params['batch_size']
        else:
            batch_size = 10
        if 'n_trees' in method_params.keys():
            n_trees=method_params['n_trees']
        else:
            n_trees=8
        if 'max_candidates' in method_params.keys():
            max_candidates=method_params['max_candidates']
        else:
            max_candidates=20
        if 'parallel_batch_queries' in method_params.keys():
            parallel_batch_queries=method_params['parallel_batch_queries']
        else:
            parallel_batch_queries = True
        if 'eps' in method_params.keys():
            eps=method_params['eps']
        else:
            eps =(0.1)**12
        if 'pow' in method_params.keys():
            pow=method_params['pow']
        else:
            pow = 2
    return N, exp, metric, max_k, c, ensure_coverage, batch_size, n_trees, max_candidates, parallel_batch_queries, eps, pow

class ContinuousInformationWeightTransformer(BaseEstimator, TransformerMixin):
    """TBD. Describe workflow.
    TODO: Currently have "Gaussian PDF"-based kernels. Add "Gaussian CDF"-based kernels (i.e. many weights close to 1).

    Parameters
    ----------
    The following parameters are used within the current class:

    simple_base_IWT: bool (optional, default = False)
        If false, this uses the existing information weight transformer on the document-kernel matrix. 
        Otherwise, this uses a simplified method that works well for dense matrices.

    The following parameters exist only to be passsed to the original information weight transformer:

    prior_strength: float (optional, default=1e-4)
        How strongly to weight the prior when doing a Bayesian update to
        derive a model based on observed counts of a column.

    approx_prior: bool (optional, default=True)
        Whether to approximate weights based on the Bayesian prior or perform
        exact computations. Approximations are much faster, especially for very
        large or very sparse datasets.

    supervision_weight: float (optional, default=0.95)
        Controls weight given to labels in supervised information weight transformer.

    weight_power: float (optional, default=2.0)
        Controls power used with supervised information weight transformer.

    

    Attributes
    ----------

    information_weights_: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.

    kernels_: 
    """

    def __init__(
        self,
        simple_base_IWT = False,
        prior_strength=1e-4,
        approx_prior=True,
        weight_power=2.0,
        supervision_weight=0.95,
    ):
        self.simple_IWT = simple_base_IWT
        self.prior_strength = prior_strength
        self.approx_prior = approx_prior
        self.weight_power = weight_power
        self.supervision_weight = supervision_weight

    def fit(self, data, ds_mat,  method = 'local_gaussian', method_params = None, y=None, **fit_kwds):
        """Learn both the appropriate kernel and column weighting as information weights
        from the observed count data ``X``.

        Parameters
        ----------

        data: ndarray of scipy sparse matrix of shape (n_features, n_dim)
            Each token counted by "ds_mat" is vectorized with the corresponding row of the matrix "data."

        ds_mat: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The count data to be trained on. Note that, as count data all
            entries should be positive or zero.

        method: string (optional, default = 'local_gaussian')
            Name of the method for defining kernels. At the time of writing, only a default 'local_gaussian' method is implemented.

        method_params: dict (optional, default = None):
            Parameters required by the chosen method. If not "None", must match expected input for chosen method.

        Returns
        -------
        self:
            The trained model.
        """
        if not scipy.sparse.isspmatrix(data):
            data = scipy.sparse.csc_matrix(data)

        if method is 'stump':
            return self

        if method is 'local_gaussian':
            N, exp, metric, max_k, c, ensure_coverage, batch_size, n_trees, max_candidates, parallel_batch_queries, eps, pow = read_local_gaussian_params(method_params)
            
            # Get kernel weights and sentence weights (the latter is what we normally call information weights).
            self.k_weights, self.information_weights = get_cts_IWT_weights(
                data,ds_mat,N=N,exp=exp,metric=metric,max_k=max_k,c=c,ensure_coverage = ensure_coverage,batch_size = batch_size, n_trees=n_trees, max_candidates=max_candidates,parallel_batch_queries = parallel_batch_queries,eps=eps,simple_IWT=self.simple_IWT, pow=pow)
            )

    # TODO: Check to see if the following actually makes sense. For now, I'm not using the branch where y is not None.

        if y is not None:
            unsupervised_power = (1.0 - self.supervision_weight) * self.weight_power
            supervised_power = self.supervision_weight * self.weight_power

            self.information_weights_ /= np.mean(self.information_weights_)
            self.information_weights_ = np.power(
                self.information_weights_, unsupervised_power
            )

            target_classes = np.unique(y)
            target_dict = dict(
                np.vstack((target_classes, np.arange(target_classes.shape[0]))).T
            )
            target = np.array(
                [np.int64(target_dict[label]) for label in y], dtype=np.int64
            )
            self.supervised_weights_ = information_weight(
                X, self.prior_strength, self.approx_prior, target=target
            )
            self.supervised_weights_ /= np.mean(self.supervised_weights_)
            self.supervised_weights_ = np.power(
                self.supervised_weights_, supervised_power
            )

            self.information_weights_ = (
                self.information_weights_ * self.supervised_weights_
            )
        else:
            self.information_weights_ /= np.mean(self.information_weights_)
            self.information_weights_ = np.power(
                self.information_weights_, self.weight_power
            )

        return self

    def transform(self, data, ds_mat):
        """Reweight original count matrix ``ds_mat`` based on learned information weights of columns.

        Parameters
        ----------
        data: ndarray of scipy sparse matrix of shape (n_features, n_dim)
            Each token counted by "ds_mat" is vectorized with the corresponding row of the matrix "data."

        ds_mat: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The count data to be trained on. Note that, as count data all
            entries should be positive or zero.


        Returns
        -------
        result: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The reweighted data.
        """
        result = ds_mat @ scipy.sparse.diags(self.information_weights_)
        return result
