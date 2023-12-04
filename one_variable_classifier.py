import pandas as pd
import numpy as np
import timeit
from subsets import train, test

# Variáveis de minimização do erro
errors_min = 180
best_lim = 0

# Variáveis matriz de confusão
TP = 0
FP = 0
TN = 0
FN = 0

# Leitura do arquivo com parâmetros de entrada
df = pd.read_csv("python_classifier_input.csv")

# Subconjuntos de treinamento e teste
df_train = df[df['index'].isin(train)]
df_test = df[df['index'].isin(test)]


# Função de classificação de um dado vetor de entrada de um par estereoscópico
def simple_classifier(vector):
    global errors, limit, TP, TN, FP, FN
    red_angle = vector['red_angle']

   # Comparação com a classificação real e contabilização de erros
    real_class = vector[0]
    if classification == real_class:
        if real_class == 'NOK':
            TN += 1
        else:
            TP += 1
    if classification != real_class:
        errors += 1
        print(f'ERRO NA IMAGEM {vector[7]}')
        if real_class == 'NOK':
            FP += 1
        else:
            FN += 1



# Testagem de todos os valores entre 0.25 e 0.35
# (intervalo escolhido por inspeção visual do gráfico)
for limit in np.arange(0.250, 0.350, 0.001):
    errors = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    # Aplicação da função a todos os pares estereoscópicos
    df_train.apply(simple_classifier, axis=1)
    # Redefinição dos melhores resultados
    if errors < errors_min:
        errors_min = errors
        best_lim = limit

# Redefinição das variáveis para testagem do algoritmo
limit = best_lim
errors = 0
TP = 0
FP = 0
TN = 0
FN = 0

# Gravação do momento inicial de execução da classificação
t_i = timeit.default_timer()

# Classificação das imagens de teste
df_test.apply(simple_classifier, axis=1)

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

print(f"Melhor limite de clasificação: {round(best_lim, 3)}\n"
      f"VP: {TP} FN: {FN}\n"
      f"FP: {FP} VN: {TN}\n"
      f"Acurácia: {accuracy}%\n"
      f"Precisão: {precision}%\n"
      f"Sensibilidade: {sensibility}%\n"
      f"Especificidade: {specificity}%\n"
      f"Tempo de execução: {round(delta_t, 5)} segundos")
