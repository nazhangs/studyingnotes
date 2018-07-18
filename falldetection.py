# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np

camera = cv2.VideoCapture(0)
fps = camera.get(cv2.CAP_PROP_FPS)
print("******************")
print(fps)

fgbg = cv2.createBackgroundSubtractorMOG2(history=1000,varThreshold=16,detectShadows=False)

#rectshape
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,4))
#userd for dilation,erosion
kernel = np.ones((5,5),np.uint8)
countImg = 0
objectFlag = 0

while True:
    grabbed, frame = camera.read()
        
    frame = cv2.medianBlur(frame,7)

    #background substractor   
    fgmask = fgbg.apply(frame, None, -1)
    
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, es, iterations=1)
    
    ret,fgmask = cv2.threshold(fgmask,25,255,cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    
    maxX=0
    maxY=0
    minX=0
    minY=0
    for c in contours:
      
       (x,y,w,h) = cv2.boundingRect(c)
       
       cv2.rectangle(frame, (x,y), (x+w, y+h),(255,255,0),2)
       cv2.rectangle(fgmask, (x,y), (x+w, y+h),(255,255,0),2)
       if  x > maxX :
           maxX = x 
       if  y > maxY :
           maxY = y 
       if  minX == 0.00 and minY == 0.00 :
           minX = x
           minY = y
       else :
           if minX > x:
               minX = x
           if minY > y:
               minY = y
    
    cv2.rectangle(frame,(minX,maxY),(maxX,minY),(255, 255, 0),2)
    cv2.rectangle(fgmask,(minX,maxY),(maxX,minY),(255, 255, 0),2)
    
    figureWidth = maxX - minX
    figureHeight = maxY - minY
    
    if figureHeight != 0 :
        figureRatio = figureWidth/figureHeight
        if figureRatio < 1 and objectFlag == 0:
            print("object detected")
            objectFlag = 1
  
        if figureRatio > 1 and objectFlag == 1:
            print("fall")
    
    
    cv2.imshow('frame',frame)
    cv2.imshow('original',fgmask)
    
    k = cv2.waitKey(1) & 0xff
    
    if k == ord('q'):
        break
    elif k == ord('s'):
        imageName_mask = str(countImg)+'mask.png'
        imageName_original = str(countImg) + 'original.png'
        cv2.imwrite(imageName_mask,frame)
        cv2.imwrite(imageName_original,fgmask)
        countImg = countImg + 1


camera.release()
cv2.destroyAllWindows()
