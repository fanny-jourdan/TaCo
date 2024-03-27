import torch
import scipy
import numpy as np
import sklearn as sk
from scipy.sparse.linalg import svds

import abc

class Decomposition:
   @abc.abstractmethod
   def decompose(self, A):
      pass
   

class SparseSVD(Decomposition):
    def __init__(self, n_components=20):
        self.n_components = n_components

    def decompose(self, A): 
        if A.is_cuda:
            A = A.cpu()
        A = A.numpy() 
        _, p = A.shape
        v0 = np.ones(p)
        U, s, VT = svds(A, k=self.n_components, v0=v0)
        S = np.diag(s)
        W = np.dot(S, VT)
        return torch.tensor(U.copy()), torch.tensor(W.copy())
    

class SVD(Decomposition):
    def decompose(self, A): 
        U, S, VT = torch.linalg.svd(A)
        W = torch.mm(torch.diag(S), VT)
        return U, W
    

class TruncatedSVD1(Decomposition):
    def __init__(self, n_components=20):
        self.n_components = n_components

    def decompose(self, A): 
        A = A.numpy() 
        svd = sk.decomposition.TruncatedSVD(n_components=self.n_components)
        U = svd.fit_transform(A)
        S = svd.singular_values_
        V = svd.components_
        W = np.dot(np.diag(S), V)
        return torch.tensor(U.copy()), torch.tensor(W.copy())
    
    
class TruncatedSVD2(Decomposition):
    def __init__(self, n_components=20):
        self.n_components = n_components

    def decompose(self, A): 
        U, S, VT = torch.linalg.svd(A)
        U = U[:, :self.n_components]
        S = S[:self.n_components]
        VT = VT[:self.n_components, :]
        W = torch.mm(torch.diag(S), VT)
        return U, W


class PCA(Decomposition):
    def decompose(self, A): 
        A = A.numpy()
        pca = sk.decomposition.PCA()
        U = pca.fit_transform(A)
        W = pca.components_
        return torch.tensor(U.copy()), torch.tensor(W.copy())


class ICA(Decomposition):
    def __init__(self, n_components=20):
        self.n_components = n_components

    def decompose(self, A): 
        A = A.numpy() 
        ica = sk.decomposition.FastICA(n_components=self.n_components, random_state=0)
        U = ica.fit_transform(A)
        W = ica.mixing_
        W = W.T
        return torch.tensor(U.copy()), torch.tensor(W.copy())
    

class NMF(Decomposition):
    def __init__(self, n_components=20):
        self.n_components = n_components

    def decompose(self, A):
        A = A.numpy() 
        model = sk.decomposition.NMF(n_components=self.n_components, init='random', random_state=0)
        U = model.fit_transform(A)
        W = model.components_
        return torch.tensor(U.copy()), torch.tensor(W.copy())
    


def decompose_choice(method_name, n_components=None):
    """
    This function serves as a dispatcher that calls a specific decomposition method based on the provided method_name string.
    
    Parameters:
    method_name (str): The name of the decomposition method to be used. It should be one of the following strings:
        - "sSVD": Calls SparseSVD method.
        - "SVD": Calls SVD method.
        - "tSVD1,2": Calls TruncatedSVD1,2 method.
        - "PCA": Calls PCA method.
        - "ICA": Calls ICA method.
        - "NMF": Calls NMF method.
    n_components (int, optional): The number of components to be used in the decomposition methods that require it. If not provided, the default value used by the specific decomposition method will be used.
    
    Returns:
    The result of the called decomposition method or an error string if the provided method_name does not match any of the available methods.
    """
    if method_name=="sSVD":
        return(SparseSVD(n_components))
    elif method_name=="SVD":
        return(SVD())
    elif method_name=="tSVD1":
        return(TruncatedSVD1(n_components))
    elif method_name=="tSVD2":
        return(TruncatedSVD2(n_components))
    elif method_name=="PCA":
        return(PCA())
    elif method_name=="ICA":
        return(ICA(n_components))
    elif method_name=="NMF":
        return(NMF(n_components))
    else:
        return("Error: this decomposition name doesnt exist.")


