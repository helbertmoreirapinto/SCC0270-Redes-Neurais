import numpy
from neuronio import Neuronio
import numpy as np

class RNA():
    def __init__(self, camadas):
        self.camadas = []
        for c in range(1, len(camadas)):
            arr = []
            for n in range(camadas[c]):
                arr.append(Neuronio(camadas[c-1]))
            self.camadas.append(arr)
        return


    def saida(self, X):
        A = X
        for camada in self.camadas:
            A_aux = []
            for neuronio in camada:
                saida = neuronio.saida(A)
                A_aux.append(saida)
            A = A_aux
        
        return A
    

    def erro_saida(self, calculado, esperado):
        erro = []
        for i in range(len(esperado)):
            erro.append(esperado[i] - calculado[i])
        return erro
    

    def treino(self, X, Y, limite_erro=10e-5, taxa=0.5):
        # while True:
        y = self.saida(X)
        erro_saida = self.erro_saida(y, Y)
        
        # atualizar pesos
        vet = erro_saida
        camadas = self.camadas[::-1]
        for camada in camadas:            
            aux = np.zeros(len(camada[0].W))

            for i, neuronio in enumerate(camada):
                soma_pesos = np.sum(neuronio.W)
                for n in range(0, len(neuronio.W)):
                    err_ni = vet[i] * (neuronio.W[n] / soma_pesos)
                    aux[n-1] += err_ni
            
            for i, neuronio in enumerate(camada):
                neuronio.set_erro(vet[i])
            
            vet = aux
    
        return
