from matplotlib import pyplot as plt

import os
import pandas as pd
import numpy as np


def gerar_dados(n=50, num_ruido=3):
    # nivel minimo de ruido para nao ter duplicidades no dataframe de retorno
    if num_ruido <= 0:
        num_ruido = 1
    

    # modelos padrão
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

    # nomeando colunas do dataframe de retorno
    col = ['bias']              # bias
    for i in range(1, len(x[0])+1):
        col.append(f'x{i}')     # xn
    col.append('y')             # y

    # cria o dataframe de retorno
    dataframe = pd.DataFrame([],columns=col)

    # criando novo diretorio para armazenamento dos modelos com ruido
    dir = './modelos'
    subdir = [x[0] for x in os.walk(dir)]
    exec = 0 if len(subdir) == 0 else len(subdir) -1
    diretorio = f'{dir}/exec_{exec}'
    os.makedirs(diretorio)

    # gerando modelos (com ruido) a partir dos modelos padrão
    while len(dataframe) < n:
        ni = len(dataframe)

        # seleciona um modelo randomicamente
        idx_sel = np.random.randint(len(x))
        sel_x = x[idx_sel]

        # seleciona e aplica ruído ao modelo
        idx_ruido = np.random.choice(len(sel_x), size=np.random.randint(num_ruido+1), replace=False)
        for idx in idx_ruido:
            sel_x[idx] *= -1
        
        # gerando e salvando imagem do modelo com ruido
        gerar_grafico(diretorio, ni, sel_x)
        
        # insere o valor 1 ao inicio (bias)
        # insere o label resposta no fim da lista
        values = np.insert(sel_x, 0, 1)
        values = np.append(values, y[idx_sel])

        # adiciona o modelo com ruído aplicado ao dataframe de retorno
        dataframe.loc[-1] = values
        dataframe.index = dataframe.index + 1
        dataframe = dataframe.sort_index()
        dataframe = dataframe.drop_duplicates(ignore_index=True)

    return dataframe



def gerar_grafico(dir, idx_img, valores):
    plt.ioff()
    fig = plt.figure(figsize=(2,2))

    # adiciona as bolinhas pretas (valores -1 do modelo - representação do vazio)
    base_x = [[x for x in range(5)] for y in range(5)]
    base_y = [[y for x in range(5)] for y in range(5)]
    plt.scatter(base_x, base_y, color='#000000', marker='o', s=300)
    
    # adiciona as bolinhas laranja (valores 1 do modelo - representação de conteudo)
    pos_x = []
    pos_y = []
    for i, v in enumerate(valores):
        if v == 1:
            pos_x.append(int(i%5))
            pos_y.append(int(i/5))
    plt.scatter(pos_x, pos_y, color='#F4B084', marker='o', s=300)
    
    # removendo eixos
    ax=plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # salvando imagem
    plt.savefig(f'{dir}/modelo_{idx_img}.png')
    plt.close(fig)