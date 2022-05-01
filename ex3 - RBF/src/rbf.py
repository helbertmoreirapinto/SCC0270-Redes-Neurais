from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import math

# resposta do neurônio da camada de saída
def sigmoide(value):
    return 1/(1+math.exp(-value))

# resposta neurônio da camada intermediária
def gaussiana(x, w, variation):
    dif =[math.pow(xx - ww,2)  for xx, ww in zip(x,w)]
    result = sum(dif)
    return math.exp(-result/(2*variation))

class RBF():
    def __init__(self, n_saidas, n_classes, seed=None):
        # configurando seed
        if seed is not None:
            np.random.seed(seed)
        self.n_classes = n_classes
        self.pesos_saida = np.random.randn(n_saidas, n_classes+1)
        return
    
    def dist_quadr_media(self, posicao_centro,entries):
        return sum([(d-k)**2 for d,k in zip(self.centros[posicao_centro], entries)])
    
    def atualizar_pesos(self, entries, deltas):
        return [ peso + (self.eta * delta * entries) for peso, delta in zip(self.pesos_saida, deltas)]

    def calcular_camada_oculta(self, X):
        return [np.append(-1,[gaussiana(d,self.centros[c], self.variations[c]) for c in range(len(self.centros))]) for d in X]

    def calcular_saidas(self, u):
        return [sigmoide(np.dot(u, weight)) for weight in self.pesos_saida]

    def MSE(self, saidas, esperado):
        error = [ (o - d)**2 for o, d in zip(saidas, esperado)]
        return sum(error) / len(esperado)

    def teste(self, X, Y, tol_amostra:float=0.2):
        X = self.calcular_camada_oculta(X)
        saidas = [self.calcular_saidas(xi) for xi in X]
        acertos = 0
        for yi, y_i in zip(Y, saidas):
            acerto_i = 1 if max(np.abs(yi - y_i)) < (1.5)*tol_amostra else 0
            acertos += acerto_i
        return acertos / len(X)

    def treino(self, X:pd.DataFrame, Y:pd.DataFrame, max_epocas:int=200, eta:int=0.5):
        self.eta = eta
        saida_esperada = np.array([[d] for d in Y])
        kmeans = KMeans(n_clusters = self.n_classes, random_state=0).fit(X)
        resultados = [[0 for x in range(2)] for y in range(self.n_classes)]
        self.centros = kmeans.cluster_centers_

        for x in X:
            posicoes = kmeans.predict([x])
            posicao = posicoes[0]
            resultados[posicao][0] = resultados[posicao][0] + 1
            MSD = self.dist_quadr_media(posicao, x)
            resultados[posicao][1] = resultados[posicao][1] + MSD
        
        self.variations = [0 for y in range(self.n_classes)]
        for x in range(len(resultados)):
            self.variations[x] = resultados[x][1] / resultados[x][0]
        
        amostras = self.calcular_camada_oculta(X)

        epoca = 0
        result = 0
        while True:
            for i, xi in enumerate(amostras):
                while True:
                    sample_error = 0
                    saidas = self.calcular_saidas(xi)
                    sample_error = self.MSE(saidas, saida_esperada[i])
                    if(sample_error < 0.01):
                        break
                    deltas = (saida_esperada[i] - saidas) * (saidas*(np.array([1])-saidas))
                    result = self.atualizar_pesos(xi, deltas)
                    for _ in range(len(result)):
                        n = [[n for n in nw] for nw in result]
                    self.pesos_saida = n

            epoca += 1
            # print(epoca)
            if(epoca == max_epocas):
                break
