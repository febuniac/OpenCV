#!/usr/bin/python
#-*- coding:utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Adaptado da documentacao em http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
# Com a funcao drawMatches do usuario
class Kalman2D(object):  
    ''''' 
    Classe para Filtro de Kalmann 2D 
    '''  
  
    def __init__(self, processNoiseCovariance=1e-4, measurementNoiseCovariance=1e-1, errorCovariancePost=0.1):  
        ''''' 
        Constrói um novo objeto Kalman2D. O erro das covariâncias tem relações entre diferentes variáveis de estado 
        (como posição, velocidade e aceleração) em qualquer um dos modelos de transição ou covariâncias.        '''  
        
        self.kalman = cv.CreateKalman(4, 2, 0) 
        self.kalman_state = cv.CreateMat(4, 1, cv.CV_32FC1)  #convertendo a matriz(32 bit floating point signed depth in one channel )
        self.kalman_process_noise = cv.CreateMat(4, 1, cv.CV_32FC1)  #//

        self.kalman_measurement = cv.CreateMat(2, 1, cv.CV_32FC1)  #//
        print(self.kalman_measurement) 
        for j in range(4): #(j(0:3)) 
            for k in range(4):  
                self.kalman.transition_matrix[j,k] = 0  
            self.kalman.transition_matrix[j,j] = 1  

        cv.SetIdentity(self.kalman.measurement_matrix)  
  
        cv.SetIdentity(self.kalman.process_noise_cov, cv.RealScalar(processNoiseCovariance))  
        cv.SetIdentity(self.kalman.measurement_noise_cov, cv.RealScalar(measurementNoiseCovariance))  
        cv.SetIdentity(self.kalman.error_cov_post, cv.RealScalar(errorCovariancePost))  
  
        
        self.estimativas = None  
MIN_MATCH_COUNT = 10
img1 = cv2.imread('carteira_foto.jpg',0)# Imagem a procurar 
video = cv2.VideoCapture(0)# 0 para webcam
video.set(3,1024)#video quality
video.set(4,768)#video quality
  
# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT in each image
kp1, des1 = sift.detectAndCompute(img1,None)    


ret, img2 = video.read()
if (ret):
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        #color = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
        #cv2.imwrite("img2.png", gray)
kp2, des2 = sift.detectAndCompute(img2,None)
#kp2, des2 = cv2.SIFT().detectAndCompute(img2, None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

# Configura o algoritmo de casamento de features
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Tenta fazer a melhor comparacao usando o algoritmo
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>MIN_MATCH_COUNT:
    # Separa os bons matches na origem e no destino
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


    # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # Transforma os pontos da imagem origem para onde estao na imagem destino
    dst = cv2.perspectiveTransform(pts,M)#quadrado azul acha

    # Desenha as linhas
    img2b = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)
    Altura = 5.4 # cm(paquimetro)

    linha1=abs(np.int32(dst[0][0][1])-np.int32(dst[1][0][1])) # abs é modulo #ponto de baixo
    linha2=abs(np.int32(dst[2][0][1])-np.int32(dst[3][0][1]))# ponto de cima  #pegamos a altura do quadrado azul

    print(linha1)
    print(linha2)
    Altura_px = abs((linha1 + linha2 )/2)#ver no photoshop # pixels(photoshop)
    Distancia_Inicial_camera = 15.5 # cm (paquimetro) # tenho que estar nesta distancia durante o primeiro frame
    Foco = Distancia_Inicial_camera * Altura_px / Altura  # F = D*h/H (pixels)
    
else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
draw_params = dict(matchColor = (0,0,255), singlePointColor = None, matchesMask = matchesMask, flags = 2)
cv2.imshow("frame",img2)
 # Imagem do cenario - puxe do video para fazer isto


while True:
 
    ret, img2 = video.read()
    if (ret):
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            #color = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
            #cv2.imwrite("img2.png", gray)
    kp2, des2 = sift.detectAndCompute(img2,None)
    #kp2, des2 = cv2.SIFT().detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    # Configura o algoritmo de casamento de features
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:#ao achar a imagem circula ela
        # Separa os bons matches na origem e no destino
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


        # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # Transforma os pontos da imagem origem para onde estao na imagem destino
        dst = cv2.perspectiveTransform(pts,M)#quadrado azul acha

        # Desenha as linhas
        img2b = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)
        # quando achar a imagem deve calcular a distancia.
        linha1_nova=abs(np.int32(dst[0][0][1])-np.int32(dst[1][0][1])) # abs é modulo
        linha2_nova=abs(np.int32(dst[2][0][1])-np.int32(dst[3][0][1]))

        Altura_px_nova = abs((linha1_nova + linha2_nova)/2)
        print("Altura_px_nova")
        print(Altura_px_nova)
        #distancia nova
        # Distancia_nova = (Foco*Altura)/Altura_px_nova
        print("Sua Distancia é:")
#print(Distancia_nova)
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
    draw_params = dict(matchColor = (0,0,255), singlePointColor = None, matchesMask = matchesMask, flags = 2)
    cv2.imshow("frame",img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break