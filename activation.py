import numpy as np

class Sigmoid:
    def __call__(self, X):
        return self.eval(X)
    
    def eval(self, X):
        return 1/(1+np.e**(-1*X))

    def grad_input(self, X):
        return np.einsum('ij,im->mij', np.identity(X.shape[0]), self.eval(X)*(1 - self.eval(X)))