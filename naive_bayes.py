import pandas as pd
import timeit
from scipy.stats import norm, exponpow
from subsets import train, test


# Valores das distribuições de frequência de
# ângulo em escala de cinza das duas classes
def p_angle(angle, group):
    if group == 'OK':
        mean = 0.2895018588984701
        std = 0.025039528243060813
        return norm.pdf(angle, mean, std)
    else:
        mean = 0.36763085451491934
        std = 0.04110383644769726
        return norm.pdf(angle, mean, std)


# Valores das distribuições de frequência de
# correlação em escala de cinza das duas classes
def p_corr(corr, group):
    if group == 'OK':
        mean = 0.9074658887516327
        std = 0.015600000919686344
        return norm.pdf(corr, mean, std)
    else:
        b = 115.15734769239427
        mean = -2.5883521454193756
        std = 3.4678021839669566
        return exponpow.pdf(corr, b, mean, std)


# Valores das distribuições de frequência de
# ângulo do canal vermelho das duas classes
def p_red_angle(red_angle, group):
    if group == 'OK':
        mean = 0.2661754265466612
        std = 0.026883493622242606
        return norm.pdf(red_angle, mean, std)
    else:
        mean = 0.34745955020134883
        std = 0.04044488186945706
        return norm.pdf(red_angle, mean, std)


# Valores das distribuições de frequência de
# correlação do canal vermelho das duas classes
def p_red_corr(red_corr, group):
    if group == 'OK':
        mean = 0.9212786530048459
        std = 0.015770149772608207
        return norm.pdf(red_corr, mean, std)
    else:
        b = 9.832956869005034
        mean = 0.5613280523166392
        std = 0.33295798837522117
        return exponpow.pdf(red_corr, b, mean, std)


# Valores das distribuições de frequência de
# similaridade ORB em escala de cinza das duas classes
def p_gsc_orb_sim(gsc_orb_sim, group):
    if group == 'OK':
        b = 6.500217770195301
        mean = -0.1413251979194281
        std = 0.9578017169322486
        return exponpow.pdf(gsc_orb_sim, b, mean, std)
    else:
        mean = 0.4056501773884906
        std = 0.16690541923164579
        return norm.pdf(gsc_orb_sim, mean, std)


# Valores das distribuições de frequência de
# similaridade ORB do canal azul das duas classes
def p_blue_orb_sim(blue_orb_sim, group):
    if group == 'OK':
        mean = 0.671884048875628
        std = 0.12694904706902366
        return norm.pdf(blue_orb_sim, mean, std)
    else:
        mean = 0.3800335714857796
        std = 0.1527570774086768
        return norm.pdf(blue_orb_sim, mean, std)


# Variável para contabilização de erros
TP = 0
FP = 0
TN = 0
FN = 0


# Classificador Naive-Bayes
def classifier(data):
    global TP, FP, TN, FN
    vector = data[params]

    # Cálculo das distribuições de frequência da classe OK
    p1_ok = p_angle(vector['angle'], 'OK')
    p2_ok = p_corr(vector['corr'], 'OK')
    p3_ok = p_red_angle(vector['red_angle'], 'OK')
    p4_ok = p_red_corr(vector['red_corr'], 'OK')
    p5_ok = p_gsc_orb_sim(vector['gsc_orb_sim'], 'OK')
    p6_ok = p_blue_orb_sim(vector['blue_orb_sim'], 'OK')

    # Probabiliade de pertencer à classe OK
    probability_ok = p1_ok*p2_ok*p3_ok*p4_ok*p5_ok*p6_ok

    # Cálculo das distribuições de frequência da classe NOK
    p1_nok = p_angle(vector['angle'], 'NOK')
    p2_nok = p_corr(vector['corr'], 'NOK')
    p3_nok = p_red_angle(vector['red_angle'], 'NOK')
    p4_nok = p_red_corr(vector['red_corr'], 'NOK')
    p5_nok = p_gsc_orb_sim(vector['gsc_orb_sim'], 'NOK')
    p6_nok = p_blue_orb_sim(vector['blue_orb_sim'], 'NOK')

    # Probabiliade de pertencer à classe NOK
    probability_nok = p1_nok*p2_nok*p3_nok*p4_nok*p5_nok*p6_nok

    if probability_ok > probability_nok:
        classification = 'OK'
    else:
        classification = 'NOK'
    real_class = data[0]
    if classification == real_class:
        if real_class == 'NOK':
            TN += 1
        else:
            TP += 1
    if classification != real_class:
        print(f'ERRO NA IMAGEM {data[7]}')
        if real_class == 'NOK':
            FP += 1
        else:
            FN += 1
    return classification


# Parâmetros classificadores
params = ['angle',
          'corr',
          'red_angle',
          'red_corr',
          'gsc_orb_sim',
          'blue_orb_sim']

# Leitura do arquivo de entrada
df = pd.read_csv("python_classifier_input.csv")
df_test = df[df['index'].isin(test)]
df_train = df[df['index'].isin(train)]

# Gravação do momento inicial de execução da classificação
t_i = timeit.default_timer()

# Classificação
df_test.apply(classifier, axis=1)

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
