import numpy as np

class GradientDescentOptimizer:
    def __init__(self):
        pass

    def set_lr(self, learning_rate):
        self.learning_rate = learning_rate
        return self
    
    def minimize(self, X, grad_X):
        assert X.shape == grad_X.shape, f"Shape mismatch, Input shape {X.shape} != Gradient shape {grad_X.shape}"
        return X - (self.learning_rate*grad_X)
    
    def maximize(self, X, grad_X):
        assert X.shape == grad_X.shape, f"Shape mismatch, Input shape {X.shape} != Gradient shape {grad_X.shape}"
        return X + (self.learning_rate*grad_X)

        
        