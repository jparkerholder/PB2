

from GPy.kern import Kern
from GPy.core import Param
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np

class TV_SquaredExp(Kern):
    def __init__(self,input_dim, variance=1.,lengthscale=1.,epsilon=0.,active_dims=None):
        super().__init__(input_dim, active_dims, 'time_se')
        self.variance = Param('variance', variance)
        self.lengthscale = Param('lengthscale', lengthscale)
        self.epsilon = Param('epsilon', epsilon)
        self.link_parameters(self.variance, self.lengthscale, self.epsilon)
        
    def K(self,X,X2):
        # time must be in the far left column
        if self.epsilon > 0.5: # 0.5
            self.epsilon = 0.5
        if X2 is None: X2 = np.copy(X)
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)
        dists = pairwise_distances(T1,T2, 'cityblock')
        timekernel=(1-self.epsilon)**(0.5*dists)
        
        X = X[:, 1:]
        X2 = X2[:, 1:]

        RBF = self.variance*np.exp(-np.square(euclidean_distances(X,X2))/self.lengthscale)
        
        return RBF * timekernel
    
    def Kdiag(self,X):
        return self.variance*np.ones(X.shape[0])
    
    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = np.copy(X)
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)
        
        X = X[:, 1:]
        X2 = X2[:, 1:]
        dist2 = np.square(euclidean_distances(X,X2))/self.lengthscale
    
        dvar = np.exp(-np.square((euclidean_distances(X,X2))/self.lengthscale))
        dl =  - (2 * euclidean_distances(X,X2)**2 * self.variance * np.exp(-dist2)) * self.lengthscale**(-2)
        n = pairwise_distances(T1,T2, 'cityblock')/2
        deps = -n * (1-self.epsilon)**(n-1)
    
        self.variance.gradient = np.sum(dvar*dL_dK)
        self.lengthscale.gradient = np.sum(dl*dL_dK)
        self.epsilon.gradient = np.sum(deps*dL_dK)
 