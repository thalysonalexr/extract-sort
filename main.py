# Disciplina: Processamento de Imagens
# 8° semestre
# Sistemas de Informação
# Universidade Federal de Mato Grosso - UFMT
# Campus Universitario Rondonopolis - CUR
# Discentes: THALYSON A. R. DE SOUSA e WANDERSON RODRIGUES DA SILVA
######################################################################################
# Links:
# http://scikit-image.org/docs/dev/api/skimage.io.html
# https://matplotlib.org/
# http://www.numpy.org/
# http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
######################################################################################


__author__ = 'Thalyson Alexandre Rodrigues de Sousa'
__date__ = '2017-09-08'

import sys
import os.path
from os import mkdir, walk
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC
from sklearn.metrics import auc, roc_curve


IMAGES = {}
CLASSES_Y = []
index_roi1 = 3
index_roi2 = 4
DIR_FILES = 'files_csv'
CSV_CLASS = 'f-diag_UFF.csv'


def read_files():
    archives = []
    root = None

    if len(sys.argv) == 1:
        direct = input('DIR >> ')
    else:
        direct = sys.argv[1]

    if not os.path.exists(direct):
        print('Não existe o diretorio "',direct,'", operação abortada.')
        exit(1)

    for (root, directory, files) in walk(direct):
        archives.extend(files)  # nome dos arquivos e imagens
        break

    print('Quantidade de registros na pasta %s: %d' % (str(root), int(len(archives) / 3)))

    for a in range(0, len(archives), 3):
        key = archives[a][3:7]
        imgRig = io.imread(direct + '\\' + archives[a], as_grey=True)
        ImgLef = io.imread(direct + '\\' + archives[a + 1], as_grey=True)
        matrix = np.loadtxt(direct + '\\' + archives[a + 2])

        LENGHT = matrix.shape
        imgLeftNew  = np.zeros(LENGHT)
        imgRightNew = np.zeros(LENGHT)
        imgFinal    = np.zeros(LENGHT)

        for i in range(LENGHT[0]):
            for j in range(LENGHT[1]):
                if ImgLef[i, j] == 255:
                    imgLeftNew[i, j] = matrix[i, j]
                    imgFinal[i, j] = matrix[i, j]
                if imgRig[i, j] == 255:
                    imgRightNew[i, j] = matrix[i, j]
                    imgFinal[i, j] = matrix[i, j]

        imgRN = (255 * (imgRightNew - np.max(imgRightNew)) / -np.ptp(imgRightNew)).astype(int)
        imgLN = (255 * (imgLeftNew  - np.max(imgLeftNew))  / -np.ptp(imgLeftNew )).astype(int)
        imgF  = (255 * (imgFinal    - np.max(imgFinal  ))  / -np.ptp(imgFinal   )).astype(int)

        IMAGES[int(key)] = [imgRig, ImgLef, matrix, imgRN, imgLN, imgF]


def show_img(id_img):
    try:
        plt.figure(1)

        plt.subplot(231)
        plt.imshow(IMAGES[id_img][0], cmap=plt.cm.gray)
        plt.title('Mama direita')
        plt.axis('off')

        plt.subplot(232)
        plt.imshow(IMAGES[id_img][1], cmap=plt.cm.gray)
        plt.title('Mama esquerda')
        plt.axis('off')

        plt.subplot(233)
        plt.imshow(IMAGES[id_img][2], cmap=plt.cm.gray)
        plt.title('Matriz')
        plt.axis('off')

        plt.subplot(234)
        plt.imshow(IMAGES[id_img][3], cmap=plt.cm.gray)
        plt.title('Mama Direita NOVA')
        plt.axis('off')

        plt.subplot(235)
        plt.imshow(IMAGES[id_img][4], cmap=plt.cm.gray)
        plt.title('Mama Esquerda NOVA')
        plt.axis('off')

        plt.subplot(236)
        plt.imshow(IMAGES[id_img][5], cmap=plt.cm.gray)
        plt.title('Imagem FINAL')
        plt.axis('off')

        plt.show()
        plt.close()
        return ''
    except KeyError:
        return 'O ID não existe!'


def gen_features(array, angle, delta):
    for key in IMAGES:
        img_right = IMAGES[key][index_roi1][:]  # Imagens de interesse nos indices
        img_left = IMAGES[key][index_roi2][:]   #
        GLCM_right = greycomatrix(img_right, [delta], [angle], levels=256, normed=True)
        GLCM_left = greycomatrix(img_left, [delta], [angle], levels=256, normed=True)

        array[key] = [
            greycoprops(GLCM_right, 'contrast')[0, 0],
            greycoprops(GLCM_right, 'dissimilarity')[0, 0],
            greycoprops(GLCM_right, 'homogeneity')[0, 0],
            greycoprops(GLCM_right, 'ASM')[0, 0],
            greycoprops(GLCM_right, 'energy')[0, 0],
            greycoprops(GLCM_right, 'correlation')[0, 0],
            greycoprops(GLCM_left, 'contrast')[0, 0],
            greycoprops(GLCM_left, 'dissimilarity')[0, 0],
            greycoprops(GLCM_left, 'homogeneity')[0, 0],
            greycoprops(GLCM_left, 'ASM')[0, 0],
            greycoprops(GLCM_left, 'energy')[0, 0],
            greycoprops(GLCM_left, 'correlation')[0, 0]]


def gen_files(array, file, directory, spc=';'):
    archive = open((directory+'\\'+file), 'w', encoding='utf-8')

    for key in array:
        archive.write(str(key)+spc)
        for data in array[key]:
            archive.write(str(data)+';')
        archive.write('\n')
    archive.close()


def insert_class(file_class, array):
    archive= open(file_class, 'r', encoding='utf-8')

    for line in archive.readlines():
        data = line.split(';')
        _id = int(data[0])
        value = int(data[1])
        for key in array:
            if _id == key:
                array[key].append(value)
                break
    archive.close()


def genY(file_class):
    # carregar Y
    global CLASSES_Y
    arc = open(file_class, 'r', encoding='utf-8')
    for line in arc.readlines():
        CLASSES_Y.append(int(line.split(';')[1]))
    arc.close()


def write_file_features():

    global DIR_FILES, CSV_CLASS

    CARAC_90_1 = {}
    CARAC_90_2 = {}
    CARAC_90_3 = {}
    CARAC_135_1 = {}
    CARAC_135_2 = {}
    CARAC_135_3 = {}

    read_files()
    # 90°
    gen_features(CARAC_90_1, np.pi / 2, 1)
    gen_features(CARAC_90_2, np.pi / 2, 2)
    gen_features(CARAC_90_3, np.pi / 2, 3)
    # 135°
    gen_features(CARAC_135_1, 3 * np.pi / 4, 1)
    gen_features(CARAC_135_2, 3 * np.pi / 4, 2)
    gen_features(CARAC_135_3, 3 * np.pi / 4, 3)

    insert_class(CSV_CLASS, CARAC_90_1)
    insert_class(CSV_CLASS, CARAC_90_2)
    insert_class(CSV_CLASS, CARAC_90_3)
    insert_class(CSV_CLASS, CARAC_135_1)
    insert_class(CSV_CLASS, CARAC_135_2)
    insert_class(CSV_CLASS, CARAC_135_3)

    if not os.path.exists(DIR_FILES):
        mkdir(DIR_FILES)

    gen_files(CARAC_90_1,  'CARACT_90_1.csv',  DIR_FILES, ';')
    gen_files(CARAC_90_2,  'CARACT_90_2.csv',  DIR_FILES, ';')
    gen_files(CARAC_90_3,  'CARACT_90_3.csv',  DIR_FILES, ';')
    gen_files(CARAC_135_1, 'CARACT_135_1.csv', DIR_FILES, ';')
    gen_files(CARAC_135_2, 'CARACT_135_2.csv', DIR_FILES, ';')
    gen_files(CARAC_135_3, 'CARACT_135_3.csv', DIR_FILES, ';')

    '''
    print('CARAC 90 1\n')
    for key in CARAC_90_1:
        print('Paciente ID: %d Dados:' % key, CARAC_90_1[key])
    print('\n')

    print('CARAC 90 2\n')
    for key in CARAC_90_1:
        print('Paciente ID: %d Dados:' % key, CARAC_90_2[key])
    print('\n')

    print('CARAC 90 3\n')
    for key in CARAC_90_1:
        print('Paciente ID: %d Dados:' % key, CARAC_90_3[key])
    print('\n')

    print('CARAC 135 1\n')
    for key in CARAC_90_1:
        print('Paciente ID: %d Dados:' % key, CARAC_135_1[key])
    print('\n')

    print('CARAC 135 2\n')
    for key in CARAC_90_1:
        print('Paciente ID: %d Dados:' % key, CARAC_135_2[key])
    print('\n')

    print('CARAC 135 3\n')
    for key in CARAC_90_1:
        print('Paciente ID: %d Dados:' % key, CARAC_135_3[key])

    print('\n')
    '''


def classifier(directory, file):
    arc = open(directory+'\\'+file, 'r', encoding='utf-8')
    lst_aux = []
    for line in arc.readlines():
        lst_aux.append(list(map(np.float64, line.split(';')[1:-2])))
    arc.close()

    X_train, X_test, y_train, y_test = train_test_split(np.array(lst_aux), CLASSES_Y, test_size=.3)

    clf = NuSVC()
    clf.fit(np.array(X_train), np.array(y_train))

    pred = clf.predict(X_test)

    print(pred)

    # fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=2)


def main():
    write_file_features()
    genY(CSV_CLASS)
    classifier('files_csv', 'CARACT_90_1.csv')
    classifier('files_csv', 'CARACT_90_2.csv')
    classifier('files_csv', 'CARACT_90_3.csv')
    classifier('files_csv', 'CARACT_135_1.csv')
    classifier('files_csv', 'CARACT_135_2.csv')
    classifier('files_csv', 'CARACT_135_3.csv')


main()