#!/usr/bin/python
#-*- coding:utf-8 -*-

'''
    This example illustrates how to use Hough Transform to find lines
    Usage:
    houghlines.py [<image_name>]
    image argument defaults to ./pic1.png
    '''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np
import sys
import math

if __name__ == '__main__':
    print(__doc__)
    
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = "./pic1.png"
    cap = cv2.VideoCapture('hall_box_battery.mp4')


while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
        
        
    if(ret):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cv2.imwrite("frame.png", gray)
        
        print("1")#teste
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    src = cv2.imread("frame.png")
    
    dst = cv2.Canny(src, 100, 200) # aplica o detector de bordas de Canny à imagem src
    
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR) # Converte a imagem para BGR para permitir desenho colorido
    # Our operations on the frame come here
    # Display the resulting frame
    cv2.imshow('frame',frame)
    # Faz uma linha ligando o ponto inicial ao ponto final, com a cor vermelha (BGR)
    if True: # HoughLinesP
        lines = cv2.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 80, 10)
        a,b,c = lines.shape
        for i in range(a):
        
            green = (0,255,0)
            red = (0,0,255)
        
            r1x1 = lines[i][0][0]
            r1y1 = lines[i][0][1]
            r1x2 = lines[i][0][2]
            r1y2 = lines[i][0][3]
            r2x1 = lines[i][1][0]
            r2y1 = lines[i][1][1]
            r2x2 = lines[i][1][2]
            r2y2 = lines[i][1][3]
                
                
            cv2.line(cdst, (r1x1,r1y1) , (r1x2,r2y2) , red, 4, cv2.CV_AA)
            cv2.line(cdst, (r2x1,r2y2), (r2x2,r2y2), red, 4, cv2.CV_AA)
            #media
            cv2.line(cdst, ((r1x1+r1x2+r2x1+r2x2)/4, (r1y1+r1y2+r2y1+r2y2)/4),\
            ((r1x1+r1x2+r2x1+r2x2)/4, 0), green, 5, cv2.CV_AA)        
            
            #tentaiva de media
            #cv2.line(cdst, (r1x1-r1x2, r1y1-r1y2), (r2x1-r2x2, r2y1-r2y2), green, 5, cv2.CV_AA)
                
            cv2.imshow("Video", cdst)
        #cv2.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.CV_AA)
    else:
        end_streaming(cap)
else:    # HoughLines
    # Esperamos nao cair neste caso
    lines = cv2.HoughLines(dst, 1, math.pi/180.0, 50, np.array([]), 0, 0)
    a,b,c = lines.shape
    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = a*rho, b*rhos
        pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
        pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
        cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.CV_AA)
    print("Used old vanilla Hough transform")
    print("Returned points will be radius and angles")

cv2.imshow("source", src)
cv2.imshow("detected lines", cdst)
cv2.waitKey(200)


