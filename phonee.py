#import numpy as np
#import cv2
#cap = cv2.VideoCapture()
#cap.open("http://10.92.51.64:8080/")
#
#while(True):
    # Capture frame-by-frame
   # ret, frame = cap.read(0)
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, 0)
    # Display the resulting frame
    #cv2.imshow('frame',gray)
   # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
# When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()
from pylab import *
from numpy import *
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import cv
import base64
import time
import urllib2

from cv2 import __version__

print(__version__)
video="http://10.92.51.64:8080/"
vv = cv.CaptureFromFile(video)
cv2.namedWindow("preview")
vc = cv2.VideoCapture("http://10.92.51.64:8080/")