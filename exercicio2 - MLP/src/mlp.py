from numpy import exp, random
import numpy as np

def sigmoide(z):
    return 1 / (1 + exp(-z))


def dsigmoide(z):
    return sigmoide(z) * (1 - sigmoide(z))


def funcao_custo(a, b):
    return a - b

class MLP():
    def __init__(self, camadas):
        self.n_camadas = len(camadas)
        self.pesos = [2*random.rand(b,a)-1 for a, b in zip(camadas[1:], camadas[:-1])]
        self.bias = [2*random.rand(1,a)-1 for a in camadas[1:]]
        return
    
    # dado um vetor de entrada, calcula a saida
    def feedforward(self, u, fa=sigmoide):
        a = [np.array(u)]
        for peso, bias in zip(self.pesos, self.bias):
            u = fa(np.dot(u, peso) + bias)
            a.append(u)
        return u, a
    
    # calcula os deltas e o nablaiente dos pesos
    def backpropagation(self, amostra, taxa, fa=sigmoide, dfa=dsigmoide):
        x, y = amostra
        
        saida, a = self.feedforward(x, fa)
        erro = funcao_custo(y, saida)

        # ajustar pesos e bias
        delta = [erro * dfa(saida)]
        for i in range(len(a)-1, 0, -1):
            nabla = delta[len(delta)-1].dot(self.pesos[i-1].T) * dfa(a[i-1])
            delta.append(nabla)
        delta = delta[::-1][1:]
        
        for i, w in enumerate(self.pesos):
            if len(a[i].shape) == 1:
                a[i] = a[i].reshape(1, len(a[i]))
            w += (taxa * np.dot(a[i].T, delta[i]))

        for i, b in enumerate(self.bias):
            b += (taxa * delta[i])
        
    
    # dado um conjunto de treino, ajusta os pesos
    def treino(self, amostras, taxa=0.5, lim_epocas=100):
        for epoca in range(lim_epocas):
            mse = 0
            for amostra in amostras:
                x, y = amostra
                saida, _ = self.feedforward(x)
                mse += np.sum(funcao_custo(y, saida)**2)
                self.backpropagation(amostra, taxa=taxa)
            
            if epoca % 1000 == 0:
                print('Epoca: {:04d} | MSE: {:.3f}'.format(epoca, mse))
        return