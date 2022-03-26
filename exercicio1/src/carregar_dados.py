import pandas as pd
import os

def load_data():
    # define diretorio onde encontran-se as amostras
    diretorio = 'modelos/'
    val = []
    
    # carrega os dados presentes nos arquivos do diretorio em um dataframe
    for p, _, files in os.walk(os.path.abspath(diretorio)):
        for file in files:
            data = pd.read_csv(f'{diretorio}{file}', sep=" ", header=None)
            
            # bias
            arr = [1]
            for col in data:
                for lin in data.iloc[col]:
                    # x[n]
                    arr.append(lin)
            # y
            arr.append(1 if file[0]=='y' else -1)
            val.append(arr)

    # nomear colunas no dataframe de retorno
    col = ['bias']
    for c in [str(f'x{i}') for i in range(1, len(val[0])-1)]:
        col.append(c)
    col.append('y')
    
    values = pd.DataFrame(val)
    values.columns = col

    return values
