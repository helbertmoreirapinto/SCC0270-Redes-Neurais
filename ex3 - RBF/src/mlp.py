import numpy as np


def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def dsigmoide(z):
    return z * (1 - z)

def diferenca_vetorial(a, b):
    return a - b


# classe que cria uma rede de perceptrons multicamadas
class MLP():
    # camadas:list contem o numero de neuronios por camada
    def __init__(self, camadas, seed=None):
        # configurando seed
        if seed is not None:
            np.random.seed(seed)
        
        # inicia randomicamente (media=0 e dp=1) os pesos dos neuronios e dos bias
        self.pesos = [np.random.randn(b,a) for a, b in zip(camadas[1:], camadas[:-1])]
        self.bias = [np.random.randn(1,a) for a in camadas[1:]]
        
        # iniciando pesos anteriores com zeros
        self.pesos_ant = [np.zeros((b,a)) for a, b in zip(camadas[1:], camadas[:-1])]
        self.bias_ant = [np.zeros((1,a)) for a in camadas[1:]]
        return
    
    # dado um vetor de entrada calcula a saida
    def feedforward(self, u, fa=sigmoide):
        # array A armazena os valores das saidas das camadas
        A = [np.array(u)]
        for peso, bias in zip(self.pesos, self.bias):
            u = fa(np.dot(u, peso) + bias)
            A.append(u)
        return u, A
    

    # propaga o erro da saida para as camadas iniciais para ajustar os pesos da rede minimizando a funcao de custo
    def backpropagation(self, x, y, taxa, alpha, fa=sigmoide, dfa=dsigmoide, funcao_custo=diferenca_vetorial):
        saida, a = self.feedforward(x, fa)
        erro = funcao_custo(y, saida)

        # ajustar pesos e bias
        # calcular os deltas da camada de saida para a camada de entrada
        delta = [erro * dfa(saida)]
        for i in range(len(a)-1, 0, -1):
            delta_i = delta[len(delta)-1].dot(self.pesos[i-1].T) * dfa(a[i-1])
            delta.append(delta_i)
        # revertendo os deltas
        delta = delta[::-1][1:]
        
        # calculando os gradientes e aplicando alterações nos pesos
        # w = w + (taxa * (entrada . delta))
        for i, w in enumerate(self.pesos):
            # bug da unidimensionalidade
            if len(a[i].shape) == 1:
                a[i] = a[i].reshape(1, len(a[i]))
            grad_p = (taxa * np.dot(a[i].T, delta[i]))
            w += grad_p + (alpha * self.pesos_ant[i])
            self.pesos_ant[i] = grad_p
        
        # aplicando alterações nos bias
        # b = b + (taxa * (1 . delta))
        for i, b in enumerate(self.bias):
            grad_b = (taxa * delta[i])
            b += grad_b + (alpha * self.bias_ant[i])
            self.bias_ant[i] = grad_b
        return

    
    # dado um conjunto de treino, ajusta os pesos
    def treino(self, X, Y, max_epocas=1000, taxa=0.5, alpha=0.5, funcao_custo=diferenca_vetorial):
        errolist = []

        for epoca in range(max_epocas):
            erro_epoca = []

            for x, y in zip(X, Y):
                saida, _ = self.feedforward(x)
                erro_epoca.append(funcao_custo(y, saida))
                self.backpropagation(x, y, taxa=taxa, alpha=alpha)
            
            # erro quadratico medio
            mse = np.sum(np.square(erro_epoca)) / len(erro_epoca)
            errolist.append(mse)
            
            # a cada 5mil epocas exibe o quadratico medio da epoca
            # if epoca % 5000 == 0:
            #     print('Epoca: {:05d} | MSE: {:.8f}'.format(epoca, mse))

        # print('Epoca: {:05d} | MSE: {:.8f}'.format(epoca, mse))
        return errolist


    def teste(self, X, Y, funcao_custo=diferenca_vetorial, tol_amostra:float=0.2):
        acertos = 0
        for x, y in zip(X, Y):
            saida, _ = self.feedforward(x)
            acerto_i = 1 if max(np.abs(saida-y)) < tol_amostra else 0
            acertos += acerto_i
        return acertos/len(X)