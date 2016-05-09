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
  

def inicio():
    cap = cv2.VideoCapture('hall_box_battery.mp4') 
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        print("0")#teste
                
        if(ret):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            cv2.imwrite("frame.png", gray)
            
            print("1")#teste
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("2")#teste
            break
        src = cv2.imread("frame.png")
                    
        dst = cv2.Canny(src, 50, 200) # aplica o detector de bordas de Canny à imagem src
                    
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR) # Converte a imagem para BGR para permitir desenho colorido
                    # Our operations on the frame come here
                    # Display the resulting frame
        cv2.imshow('frame',frame)
                    
        lines = cv2.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
        print (lines[0][2])
        # esta abrindo o video até aqui (tudo OK)
        print("3")
        if len(lines[0]) > 1:
                    first  = lines[0][0]
                    second = lines[0][2]
                    x1 = 0
                    y1 = 1
                    x2 = 2
                    y2 = 3
                    green = (0,255,0)
                    red = (0,0,255)

                    print("potato")
                    
                    cv2.line(cdst, (first[x1], first[y1]), (first[x2], first[y2]), red, 5, cv2.CV_AA)
                    cv2.line(cdst, (second[x1], second[y1]), (second[x2], second[y2]), red, 5, cv2.CV_AA)
                    cv2.line(cdst, (first[x1]-second[x1], first[y1]-second[y1]), (first[x2]-second[x2], first[y2]-second[y2]), green, 5, cv2.CV_AA)
                    cv2.imshow("Video", cdst)
                    if cv2.waitKey(1) == 27:
                        end_streaming(cap)

        else:
                end_streaming(cap)
        #for i in range(a):
            # Faz uma linha ligando o ponto inicial ao ponto final, com a cor vermelha (BGR)
            #cv2.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.CV_AA)

    else:    # HoughLines
        # Esperemos nao cair neste caso
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
        cv2.waitKey(0)
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    inicio()






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
import numpy as np
import cv2
import sys
import math
import time

def end_streaming(capture):
    cap.release()
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture("hall_box_battery.mp4")

    while(cap.isOpened()):

        ret, frame = cap.read()

        if(ret):

            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("frame.png", grayscale_frame)

            src = cv2.imread("frame.png")
            dst = cv2.Canny(src, 50, 200)
            cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

            lines = cv2.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
            
            if len(lines[0]) > 1:
                first  = lines[0][0]
                second = lines[0][2]
                x1 = 0
                y1 = 1
                x2 = 2
                y2 = 3
                green = (0,255,0)
                red = (0,0,255)

                cv2.line(cdst, (first[x1], first[y1]), (first[x2], first[y2]), red, 5, cv2.CV_AA)
                cv2.line(cdst, (second[x1], second[y1]), (second[x2], second[y2]), red, 5, cv2.CV_AA)
                cv2.line(cdst, (first[x1]-second[x1], first[y1]-second[y1]), (first[x2]-second[x2], first[y2]-second[y2]), green, 5, cv2.CV_AA)


            cv2.imshow("Video", cdst)
            if cv2.waitKey(1) == 27:
                end_streaming(cap)

        else:
            end_streaming(cap)

if __name__ == '__main__':
    main()  
