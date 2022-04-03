from matplotlib import pyplot as plt

import os
import pandas as pd
import numpy as np


def gerar_dados(n=50, num_ruido=3):
    # nivel minimo de ruido para nao ter duplicidades no dataframe de retorno
    if num_ruido <= 0:
        num_ruido = 1
    
    # modelos padrão
    modelos_padrao = []
    modelos_padrao.append(np.array([
        [+1, -1, -1, -1, +1],
        [+1, -1, -1, -1, +1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, +1, +1, +1, -1]
    ]))
    modelos_padrao.append(np.array([
        [+1, -1, -1, -1, +1],
        [-1, +1, -1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, +1, +1, +1, -1]
    ]))
    modelos_padrao.append(np.array([
        [+1, -1, -1, -1, +1],
        [-1, +1, -1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]))
    modelos_padrao.append(np.array([
        [+1, -1, -1, -1, +1],
        [+1, +1, -1, +1, +1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]))
    modelos_padrao.append(np.array([
        [-1, +1, -1, +1, -1],
        [-1, +1, -1, +1, -1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]))
    modelos_padrao.append(np.array([
        [+1, +1, -1, +1, +1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, +1, +1, +1, -1]
    ]))
    modelos_padrao.append(np.array([
        [+1, -1, -1, -1, +1],
        [+1, -1, -1, -1, +1],
        [+1, +1, +1, +1, +1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]))
    modelos_padrao.append(np.array([
        [+1, -1, -1, -1, +1],
        [+1, -1, -1, -1, +1],
        [-1, +1, -1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]))
    modelos_padrao.append(np.array([
        [-1, +1, -1, +1, -1],
        [-1, +1, -1, +1, -1],
        [-1, +1, -1, +1, -1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1]
    ]))
    modelos_padrao.append(np.array([
        [-1, -1, -1, -1, -1],
        [+1, +1, -1, +1, +1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, +1, +1, +1, -1]
    ]))
    modelos_padrao.append(np.array([
        [+1, +1, -1, +1, +1],
        [-1, +1, -1, +1, -1],
        [-1, +1, +1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]))
    modelos_padrao.append(np.array([
        [-1, +1, -1, +1, -1],
        [-1, +1, -1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, -1, -1, -1]
    ]))
    
    # adiciona os modelos de Y invertido utilizando o comando flip
    for m in range(len(modelos_padrao)):
        modelos_padrao.append(np.flip(modelos_padrao[m]))
    
    # transforma as matrizes dos modelos em listas utilizando o comando flatten
    # cria uma lista com os valores esperados (variavel resposta)
    modelos_padrao = [modelo_padrao.flatten() for modelo_padrao in modelos_padrao]
    y = [1 if i < len(modelos_padrao)/2 else -1 for i in range(len(modelos_padrao))]

    # salvando imagens dos modelos padrão
    for modelo_padrao_i, modelo_padrao in enumerate(modelos_padrao):
        gerar_grafico('./modelos', modelo_padrao_i, modelo_padrao)
    
    # nomeando colunas do dataframe de retorno
    col = ['bias']              # bias
    for i in range(1, len(modelos_padrao[0])+1):
        col.append(f'x{i}')     # xn
    col.append('y')             # y

    # cria o dataframe de retorno
    dataframe = pd.DataFrame([],columns=col)

    # criando novo diretorio para armazenamento dos modelos gerados com ruido
    dir = './gerados'
    subdir = [p[0] for p in os.walk(dir)]
    exec = 0 if len(subdir) == 0 else len(subdir) -1
    diretorio = f'{dir}/exec_{exec}'
    os.makedirs(diretorio)

    # gerando modelos (com ruido) a partir dos modelos padrão
    while len(dataframe) < n:
        ni = len(dataframe)

        # seleciona um modelo randomicamente
        indice_modelo = np.random.randint(len(modelos_padrao))
        modelo_escolhido = modelos_padrao[indice_modelo].copy()

        # seleciona e aplica ruído ao modelo
        array_indices_ruido = np.random.choice(len(modelo_escolhido), size=1+np.random.randint(num_ruido), replace=False)
        for indice in array_indices_ruido:
            modelo_escolhido[indice] *= -1
        
        # gerando e salvando imagem do modelo com ruido
        str_arr_ruido = '_'.join([str(a) for a in array_indices_ruido])
        nome_img = f'img_{ni}_modelo_{indice_modelo}_ruido_{str_arr_ruido}'
        gerar_grafico(diretorio, nome_img, modelo_escolhido)
        
        # insere o valor 1 ao inicio (bias)
        # insere o label resposta no fim da lista
        modelo_ruido = np.insert(modelo_escolhido, 0, 1)
        modelo_ruido = np.append(modelo_ruido, y[indice_modelo])

        # adiciona o modelo com ruído aplicado ao dataframe de retorno
        dataframe.loc[-1] = modelo_ruido
        dataframe.index = dataframe.index + 1
        dataframe = dataframe.sort_index()
        dataframe = dataframe.drop_duplicates(ignore_index=True)

    return dataframe



def gerar_grafico(dir, nome_img, valores):
    plt.ioff()
    fig = plt.figure(figsize=(5,5))

    # adiciona as bolinhas pretas (valores -1 do modelo - representação do vazio)
    base_x = [[x for x in range(5)] for y in range(5)]
    base_y = [[y for x in range(5)] for y in range(5)]
    plt.scatter(base_y, base_x, color='#FFFFFF', marker='s', s=2250)
    
    # adiciona as bolinhas laranja (valores 1 do modelo - representação de conteudo)
    pos_x = []
    pos_y = []
    for i, v in enumerate(valores):
        if v == 1:
            pos_x.append(4 - int(i/5))
            pos_y.append(4 - int(i%5))
    plt.scatter(pos_y, pos_x, color='#000000', marker='s', s=2250)
    
    # removendo eixos
    ax=plt.gca()
    plt.axis([-1, 5, -1, 5])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # salvando imagem
    plt.savefig(f'{dir}/{nome_img}.png')
    plt.close(fig)