import numpy as np

class Perceptron():
    def __init__(self, n_entradas, tx=0.5):
        # pesos para bias + entradas
        self.w = 2*np.random.rand(1 + n_entradas) - 1
        self.tx = tx
        self.epoca = 0

    # degrau bipolar
    def funcao_ativacao(self, u):
        return -1 if u <= 0 else 1
    
    # dado array de entradas faz uma previsao de classificacao
    def prever(self, x):
        u = np.dot(x, self.w)
        return self.funcao_ativacao(u)

    # algoritmo de atualizacao de pesos da RNA
    def treinar(self, data):
        
        # algoritmo so acaba qnd o erro em todas as amostras de treino = 0
        while True:
            erro_epoca = 0
            for i in data.index:
                # X = variaveis explicativas
                # Y = variavel resposta para instancia
                X = [x for x in data.loc[i, data.columns != 'y']]
                Y = data['y'][i]
                
                # continua enquanto nao prever corretamente a instancia atual
                while True:
                    # y = previsao do amostra atual
                    # Y = valor esperado
                    y = self.prever(X)
                    erro_amostra = Y - y
                    if erro_amostra == 0:
                        break
                        
                    # se valor previsto != esperado, atualiza os pesos
                    # delta w = taxa aprendizagem * erro de previsao * vetor x de entradas
                    delta_w = np.dot((self.tx * erro_amostra), X)
                    self.w += delta_w
                    erro_epoca += abs(erro_amostra)
            
            if erro_epoca == 0:
                break

            self.epoca += 1
        