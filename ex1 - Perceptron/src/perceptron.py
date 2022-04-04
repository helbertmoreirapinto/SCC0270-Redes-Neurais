import numpy as np

class Perceptron():
    def __init__(self, n_entradas, seed, tx=0.5):
        np.random.seed(seed)

        # inicia randomicamente [-1, 1] os pesos para bias + entradas
        self.w = 2*np.random.rand(1 + n_entradas)-1

        # taxa de aprendizagem
        self.tx = tx
        
        # indica quantas vezes foi necessario processar o conjunto completo dos dados
        # ate que os pesos sejam ajustados para acertar a previsao em todas as amostras
        self.epoca = 0


    # degrau bipolar
    def funcao_ativacao(self, u):
        return -1 if u <= 0 else 1
    

    # dado um array de entradas faz uma previsao de classificacao
    def prever(self, x):
        u = np.dot(x, self.w)
        return self.funcao_ativacao(u)


    # função que avalia se saida é igual a desejada
    def comparar(self, calculado, processado):
        return calculado - processado

    
    # algoritmo de atualizacao de pesos da RNA
    def treinar(self, data):
        # algoritmo so acaba qnd o erro em todas as amostras de treino = 0
        while True:
            self.epoca += 1
            erro_epoca = 0
            
            for i in data.index:
                # X = variaveis explicativas
                # Y = variavel resposta da instancia
                X = [x for x in data.loc[i, data.columns != 'y']]
                Y = data['y'][i]
                
                # continua enquanto nao prever corretamente a instancia atual
                while True:
                    # y = previsao realizada para amostra atual
                    # Y = valor esperado
                    y = self.prever(X)
                    erro_amostra = self.comparar(Y, y)
                    if erro_amostra == 0:
                        break
                        
                    # se valor previsto != esperado, atualiza os pesos
                    # delta w = taxa aprendizagem * erro de previsao * vetor de entradas
                    delta_w = np.dot((self.tx * erro_amostra), X)
                    self.w += delta_w
                    erro_epoca += abs(erro_amostra)
            
            if erro_epoca == 0:
                break

        