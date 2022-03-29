import pandas as pd
import numpy as np


def gerar_dados(n=50, num_ruido=3):
    x = []

    x.append(np.array([
        [+1, -1, -1, -1, +1],
        [+1, -1, -1, -1, +1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, +1, +1, +1, -1]
    ]))

    x.append(np.array([
        [+1, -1, -1, -1, +1],
        [-1, +1, -1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, +1, +1, +1, -1]
    ]))

    x.append(np.array([
        [+1, -1, -1, -1, +1],
        [-1, +1, -1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]))

    x.append(np.array([
        [+1, -1, -1, -1, +1],
        [+1, +1, -1, +1, +1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]))

    x.append(np.array([
        [-1, +1, -1, +1, -1],
        [-1, +1, -1, +1, -1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]))

    x.append(np.array([
        [+1, +1, -1, +1, +1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, +1, +1, +1, -1]
    ]))

    x.append(np.array([
        [+1, -1, -1, -1, +1],
        [+1, -1, -1, -1, +1],
        [+1, +1, +1, +1, +1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]))

    x.append(np.array([
        [+1, -1, -1, -1, +1],
        [+1, -1, -1, -1, +1],
        [-1, +1, -1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]))

    x.append(np.array([
        [-1, +1, -1, +1, -1],
        [-1, +1, -1, +1, -1],
        [-1, +1, -1, +1, -1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1]
    ]))

    x.append(np.array([
        [-1, -1, -1, -1, -1],
        [+1, +1, -1, +1, +1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, +1, +1, +1, -1]
    ]))

    x.append(np.array([
        [+1, +1, -1, +1, +1],
        [-1, +1, -1, +1, -1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]))

    x.append(np.array([
        [-1, +1, -1, +1, -1],
        [-1, +1, -1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, -1, -1, -1]
    ]))

    # adiciona os modelos de Y invertido utilizando o comando flip
    for i in range(len(x)):
        x.append(np.flip(x[i]))
    
    # transforma as matrizes dos modelos em listas utilizando o comando flatten
    # cria uma lista com os valores esperados (variavel resposta)
    x = [xi.flatten() for xi in x]
    y = [1 if i < len(x)/2 else -1 for i in range(len(x))]

    col = ['bias']
    for i in range(1, len(x[0])+1):
        col.append(f'x{i}')
    col.append('y')

    # cria o dataframe de retorno
    dataframe = pd.DataFrame([],columns=col)

    for ni in range(n):
        # seleciona um modelo randomicamente
        idx_sel = np.random.randint(len(x))
        sel_x = x[idx_sel]

        # seleciona e aplica ruído ao modelo
        idx_ruido = np.random.choice(len(sel_x), size=np.random.randint(num_ruido+1), replace=False)
        for idx in idx_ruido:
            sel_x[idx] *= -1

        # insere o valor 1 ao inicio (bias)
        # insere o label resposta no fim da lista
        values = np.insert(sel_x, 0, 1)
        values = np.append(values, y[idx_sel])

        # adiciona o modelo com ruído aplicado ao dataframe de retorno
        dataframe.loc[-1] = values
        dataframe.index = dataframe.index + 1
        dataframe = dataframe.sort_index()

    return dataframe

