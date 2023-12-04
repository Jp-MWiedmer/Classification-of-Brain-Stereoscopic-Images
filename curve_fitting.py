import pandas as pd
from fitter import Fitter
from scipy.stats import shapiro
from subsets import train

# Leitura do arquivo com os parâmetros extraídos
df = pd.read_csv("python_classifier_input.csv")
df_train = df[df['index'].isin(train)]
# Parâmetros classificadores
params = ['angle',
          'corr',
          'red_angle',
          'red_corr',
          'gsc_orb_sim',
          'blue_orb_sim']

# Separação das classes em dataframes distintas
OK_group = df_train[(df_train.status == 'OK')][params]
NOK_group = df_train[(df_train.status == 'NOK')][params]


# Ajuste das distribuições de frequência de
# todos os parâmetros, das classes OK e NOK
i = 0
for parameter in params:
    # Clase OK
    data = OK_group[parameter].to_numpy()
    # Teste de normalidade
    stat_test, normal_test = shapiro(data)
    # Se o teste indicar normalidade, 
    # ajuste de curva por Distribuição Normal
    if normal_test > 0.05:
        f = Fitter(data, distributions=['norm'])
        f.fit()
        print(f"\nGRUPO OK - {params[i]} - "
              f"{f.get_best(method='sumsquare_error')}")
    # Se indicar não normalidade, o ajuste é feito para os dois modelos.
    # Seleciona-se a curva com menor erro quadrático
    else:
        f = Fitter(data, distributions=['exponpow', 'norm'])
        f.fit()
        print(f"\nGRUPO OK - {params[i]} - "
              f"{f.get_best(method='sumsquare_error')}")

    # Classe NOK
    data = NOK_group[parameter].to_numpy()
    # Teste de normalidade
    stat_test, normal_test = shapiro(data)
    # Se o teste indicar normalidade, 
    # ajuste de curva por Distribuição Normal
    if normal_test > 0.05:
        f = Fitter(data, distributions=['norm'])
        f.fit()
        print(f"\nGRUPO NOK - {params[i]} - "
              f"{f.get_best(method='sumsquare_error')}")
    # Se indicar não normalidade, o ajuste é feito para os dois modelos.
    # Seleciona-se a curva com menor erro quadrático
    else:
        f = Fitter(data, distributions=['exponpow', 'norm'])
        f.fit()
        print(f"\nGRUPO NOK - {params[i]} - "
              f"{f.get_best(method='sumsquare_error')}")
    i += 1
