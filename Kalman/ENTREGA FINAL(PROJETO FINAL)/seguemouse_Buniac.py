# -*- coding:utf-8 -*-
#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv  
import cv2  
import numpy as np  
from sys import exit  
class Kalman2D(object):  
    ''''' 
    Classe para Filtro de Kalmann 2D 
    '''  
  
    def __init__(self, processNoiseCovariance=1e-4, measurementNoiseCovariance=1e-1, errorCovariancePost=0.1):  
        ''''' 
        Constrói um novo objeto Kalman2D. O erro das covariâncias tem relações entre diferentes variáveis de estado 
        (como posição, velocidade e aceleração) em qualquer um dos modelos de transição ou covariâncias.        '''  
        
        self.kalman = cv.CreateKalman(4, 2, 0) 
        #print(self.kalman)
        self.kalman_state = cv.CreateMat(4, 1, cv.CV_32FC1)  #convertendo a matriz(32 bit floating point signed depth in one channel )
        #print(self.kalman_state)
        self.kalman_process_noise = cv.CreateMat(4, 1, cv.CV_32FC1)  #//
        #print(self.kalman_process_noise)
        self.kalman_measurement = cv.CreateMat(2, 1, cv.CV_32FC1)  #//
        #print(self.kalman_measurement) 
        for j in range(4): #(j(0:3)) 
            for k in range(4):  
                self.kalman.transition_matrix[j,k] = 0  
            self.kalman.transition_matrix[j,j] = 1  

        cv.SetIdentity(self.kalman.measurement_matrix)  
  
        cv.SetIdentity(self.kalman.process_noise_cov, cv.RealScalar(processNoiseCovariance))  
        cv.SetIdentity(self.kalman.measurement_noise_cov, cv.RealScalar(measurementNoiseCovariance))  
        cv.SetIdentity(self.kalman.error_cov_post, cv.RealScalar(errorCovariancePost))  
  
        
        self.estimativas = None  
  
    def update(self, x, y):  
        ''''' 
        Atualiza o filtro a partir de um valor de X e Y novo
        '''  
        cv2.putText(img, "X Real: " ,(280,50), font, 1,(255,255,255),2)
        cv2.putText(img, str(x) ,(430,50), font, 1,(255,255,255),1)
        cv2.putText(img, "Y Real: " ,(280,80), font, 1,(255,255,255),2)
        cv2.putText(img, str(y) ,(430,80), font, 1,(255,255,255),1)
        self.kalman_measurement[0, 0] = x  
        print("x_real = "+(str(x)))
        self.kalman_measurement[1, 0] = y 
        print("y_real = "+(str(y)))
  
        self.predicted = cv.KalmanPredict(self.kalman)  
        self.corrected = cv.KalmanCorrect(self.kalman, self.kalman_measurement)  
  
    def estimativa(self):  
        ''''' 
        Devolve o valor atual da estimativa de X e Y. 
        '''  
  
        return self.corrected[0,0], self.corrected[1,0]  
  
# Este atraso ira afetar o taxa de atualização do Filtro de Kalman 
atraso= 20  
  
# Dados da janela do programa 
title = 'Seguidor de Mouse utilizando Filtro de Kalman [Aperte ESC para sair]'    
  


class Mouse(object):  
    ''''' 
    Salva valores de x e y 
    '''  
  
    def __init__(self):  
  
        self.x, self.y = -1, -1 
        #print(self.x,self.y)
  
    def __str__(self):  
  
        return '%4d %4d' % (self.x, self.y)  
  
def mouseCallback(event, x, y, flags, mouse_info):  
    ''''' 
    Um objeto que atualiza com novas coordenadas X e Y  do Mouse 
    '''  
  
    mouse_info.x = x  
    mouse_info.y = y  
  
  
def drawPointer(img, center, r, g, b):  
    ''''' 
    Esta classe serve para desenhar uma cruz em coordenadas X,Y com cores em RGB 
    '''  
  
    d = 3 #tamanho
    t = 20 #grossura 
  
    color = (r, g, b)  
  
    ctrx = center[0]  
    ctry = center[1]  
  
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, cv2.CV_AA)  
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, cv2.CV_AA)  
  
def drawLines(img, points, r, g, b):  
    ''''' 
    Desenha as linhas que mostram o percurso  
    '''  
  
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))  
  
  
def imagem():  
    ''''' 
    Retorna uma imagem nova
    '''  
  
    return np.zeros((700,700,3), np.uint8)  # Cria uma imagem inteira preta
  
  
if __name__ == '__main__':  
  
  
    # cria uma imagem nova na janela   
    img = imagem()  #cria
    width=1000
    height=10
    cv2.namedWindow(title) 
    cv2.resizeWindow(title, width, height) #nome



    # TExto na tela
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'CLICK ON THE SCREEN TO START',(100,350), font, 1,(255,255,255),1)


    # cria um valor de X, Y e a informação do mouse objeto e define o retorno da janela mouse para modificá-lo
    mouse_info = Mouse()  #cria
    cv2.setMouseCallback(title, mouseCallback, mouse_info)  #define
  
    # Loop até o mouse aparecer na janela 
    while True:  
  
        if mouse_info.x > 0 and mouse_info.y > 0:  
            break  
  
        cv2.imshow(title, img)  
        if cv2.waitKey(1) == 27:  
            exit(0)  
  
  
    # Listas que pegam as as trajetorias para a posicao do mouse as estimativas de Kalman 
    medidas_p = []  
    kalman_p = []  
  
    # Cria novo filtro de Kalman2D (sua initialização ocorre com a posição inicial do mouse)   
    kalman2d = Kalman2D()  
  
    # Loop até dar escape 
    while True:  
  
        # Gera uma nova imagem 
        img = imagem()  
  
        # Pega posição atual do mouse e adiciona a trajetoria 
        measured = (mouse_info.x, mouse_info.y)  
        medidas_p.append(measured)
        #print(measured)
  
        # Atualiza o filtro com o mouse 
        kalman2d.update(mouse_info.x, mouse_info.y)

  
        # Pega a estimativa atual do filtro e adiciona a trajetoria    
        estimated = [int (c) for c in kalman2d.estimativa()]  
        kalman_p.append(estimated)  
        cv2.putText(img, "X estimado: " ,(1,50), font, 1,(255,255,255),2)
        cv2.putText(img, str(estimated[0]) ,(200,50), font, 1,(255,255,255),1)
        cv2.putText(img, "Y estimado: " ,(1,80), font, 1,(255,255,255),2)
        cv2.putText(img, str(estimated[1]) ,(200,80), font, 1,(255,255,255),1)

        print("\n")
        print("x_estimado: "+str(estimated[0]))
        print("y_estimado: "+str(estimated[1]))
        print("\n")
        #print(len(estimated))

        
  
        # Mostra a linha das trajetorias medidas e estimadas (os valores sao cores em rgb) 
        drawPointer(img, estimated,       255, 255, 0) 
        drawLines(img, kalman_p,   116, 209, 234)  
        
        drawPointer(img, measured, 255,255,255)
        drawLines(img, medidas_p, 0, 0, 255)   
          

  
        # Atraso especifico para um intervalo    
        cv2.imshow(title, img)  

        if cv2.waitKey(atraso) == 27:  
            break 