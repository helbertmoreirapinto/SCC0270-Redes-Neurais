import numpy as np
from math import exp, tanh

class Neuronio():
    def __init__(self, n):
        self.W = 2*np.random.rand(n)-1
        self.bias = 2*np.random.rand()-1

        return


    def sigmoide(self, u):
        return 1 / (1 + exp(-u))


    def tan_hip(self, u):
        return tanh(u)


    def saida(self, X):
        y = np.dot(X, self.W) + self.bias
        return self.sigmoide(y)


    def set_erro(self, erro):
        self.erro = erro