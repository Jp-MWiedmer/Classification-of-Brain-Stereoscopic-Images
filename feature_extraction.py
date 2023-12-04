import glob
import cv2
import numpy as np
import pandas

# EXTRAÇÃO DE PARÂMETROS CLASSIFICADORES


def orb_similarity(L, R):

    orb = cv2.ORB_create()

    # detecção de pontos-chave e descritores
    kp_L, desc_L = orb.detectAndCompute(L, None)
    kp_R, desc_R = orb.detectAndCompute(R, None)

    # criação do algoritmo de correspondência por força bruta
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # correspondência
    matches = bf.match(desc_L, desc_R)

    # Contagem de correspondências confiáveis,
    # com distância inferior a uma constante.
    # 55 foi o valor que obteve melhor FDR
    similar_regions = [i for i in matches if i.distance < 55]
    if len(matches) == 0:
        return 0

    # Porcentagem de correspondências confiáveis
    return len(similar_regions) / len(matches)


def angle(L, R):

    # Normalização das matrizes coluna achatadas
    unit_L = L / np.linalg.norm(L)
    unit_R = R / np.linalg.norm(R)

    # Cálculo do produto interno
    dot_product = np.dot(unit_L, unit_R)

    # A função retorna o ângulo entre os vetores
    return np.arccos(dot_product)


def calc_params(ls, ok=True):
    ind = 0
    global j
    while True:
        try:

            # Importação da imagem esquerda
            L_color = cv2.imread(ls[ind], cv2.IMREAD_COLOR)
            # Obtenção do canal vermelho (outros foram testados)
            L = L_color[:, :, 2]
            # Achatamento matricial
            L = L.flatten('F')

            # Importação da imagem direita
            R_color = cv2.imread(ls[ind+1], cv2.IMREAD_COLOR)
            # Obtenção do canal vermelho (outros foram testados)
            R = R_color[:, :, 2]
            # Achatamento matricial
            R = R.flatten('F')

            # Extração de parâmetros classificadores

            # Ângulo entre matrizes coluna para o canal vermelho
            red_angle = angle(L, R)

            # Correlação entre matrizes coluna para o canal vermelho
            red_coef = np.corrcoef(L, R)[0,1]

            # Obtenção dos canais azul das imagens esquerda e direita
            L = L_color[:, :, 0]
            R = R_color[:, :, 0]

            # Extração do parâmetro de similaridade
            # baseado no algoritmo ORB para o canal azul
            blue_orb_sim = orb_similarity(L, R)

            # Conversão das imagens para escala de cinza
            L = cv2.cvtColor(L_color, cv2.COLOR_BGR2GRAY)

            R = cv2.cvtColor(R_color, cv2.COLOR_BGR2GRAY)

            # Extração do parâmetro de similaridade
            # baseado no algoritmo ORB para escala de cinza
            gsc_orb_sim = orb_similarity(L, R)

            # Achatamento matricial
            L = L.flatten('F')
            R = R.flatten('F')

            # Ângulo entre matrizes coluna para escala de cinza
            gsc_angle = angle(L, R)

            # Correlação entre matrizes coluna para escala de cinza
            gsc_coef = np.corrcoef(L, R)[0,1]

            if ok is True:
                df.loc[j] = ['OK', gsc_angle, gsc_coef, red_angle,
                             red_coef, gsc_orb_sim, blue_orb_sim,  j]
            else:
                df.loc[j] = ['NOK', gsc_angle, gsc_coef, red_angle,
                             red_coef, gsc_orb_sim, blue_orb_sim,  j]
            ind += 2
            j += 1
        except IndexError:
            break


# Organização dos arquivos das imagens
nok_list = glob.glob("./NOK 1.1/*/*/*/*/im_*.jpg")
for file in nok_list:
    if 'C' in file.split('\\')[-1]:
        nok_list.remove(file)
nok_list.sort()
ok_list = glob.glob("./OK 1.1/*/*/*/*/im_*.jpg")
for file in ok_list:
    if 'C' in file.split('\\')[-1]:
        ok_list.remove(file)
ok_list.sort()

# Criação de DataFrame para armazenamento de dados
df = pandas.DataFrame(columns=['status', 'angle', 'corr', 'red_angle', 'red_corr',
                               'gsc_orb_sim', 'blue_orb_sim', 'index'])

j = 1

# Processamento e extração de parâmetro
calc_params(ok_list, ok=True)
calc_params(nok_list, ok=False)

# Exportação dos dados obtidos
df.to_csv("python_classifier_input.csv", index=False)
