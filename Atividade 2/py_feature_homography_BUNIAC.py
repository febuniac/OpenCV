import numpy as np
import cv2
from matplotlib import pyplot as plt

# Adaptado da documentacao em http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
# Com a funcao drawMatches do usuario


MIN_MATCH_COUNT = 10

img1 = cv2.imread('carteira_foto.jpg',0)# Imagem a procurar     
#img1.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
#img1.create(height,width,CV_8UC4);
video = cv2.VideoCapture(0)# 0 para webcam
 # Imagem do cenario - puxe do video para fazer isto
while True:
 
    ret, img2 = video.read()
    if (ret):
            print("loop ret")
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            #color = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
            cv2.imwrite("img2.png", gray)

        

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT in each image
    kp1, des1 = sift.detectAndCompute(img1,None)
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
        dst = cv2.perspectiveTransform(pts,M)

        # Desenha as linhas
        img2b = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
    cv2.imshow("frame",img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    