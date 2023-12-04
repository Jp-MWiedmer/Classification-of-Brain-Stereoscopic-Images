import pandas as pd
import numpy as np
import timeit
from subsets import test
from numpy.linalg import inv

# Leitura do arquivo com parâmetros de entrada
df = pd.read_csv("python_classifier_input.csv", index_col=False)
df_test = df[df['index'].isin(test)]
params = ['angle',
          'corr',
          'red_angle',
          'red_corr',
          'gsc_orb_sim',
          'blue_orb_sim']

# Vetor obtido através do Matlab
w_T = np.transpose([10.840394155929257,
              15.68309214066979,
              -18.00600470767011,
              -14.507747281223013,
              0.4246558571418467,
              1.4727732094116368])

TP = 0
FP = 0
TN = 0
FN = 0

# Função classificadora
def lse_classifier(vector):
   global w_T, TP, FP, TN, FN
    res = w_T.dot(vector[params])
    if res > 0:
        classification = 'OK'
    else:
        classification = 'NOK'
    real_class = vector[0]
    if classification == real_class:
        if real_class == 'NOK':
            TN += 1
        else:
            TP += 1
    if classification != real_class:
        print(f'ERRO NA IMAGEM {vector[7]}')
        if real_class == 'NOK':
            FP += 1
        else:
            FN += 1


# Gravação do momento inicial de execução da classificação
t_i = timeit.default_timer()

df_test.apply(lse_classifier, axis=1)

# Gravação do momento final da classificação
t_f = timeit.default_timer()
# Tempo de execução
delta_t = t_f - t_i
# Acurácia
accuracy = round(100 * (TP + TN) / (TP + TN + FP + FN), 2)
# Precisão
precision = round(100 * TP / (TP + FP), 2)
# Sensibilidade
sensibility = round(100 * TP / (TP + FN), 2)
# Especificidade
specificity = round(100 * TN / (TN + FP), 2)
print(f"VP: {TP} FN: {FN}\n"
      f"FP: {FP} VN: {TN}\n"
      f"Acurácia: {accuracy}%\n"
      f"Precisão: {precision}%\n"
      f"Sensibilidade: {sensibility}%\n"
      f"Especificidade: {specificity}%\n"
      f"Tempo de execução: {round(delta_t, 5)} segundos")
