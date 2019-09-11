#importation
import os
import cv2
import numpy as np


#setting paths
parentDir=os.path.abspath(os.path.join(os.getcwd(), os.pardir))
faceDetect=cv2.CascadeClassifier(parentDir+'/haarcascade_frontalface_default.xml');


ID=raw_input("Please, enter, the ID");
sampleNumber=0;
cam=cv2.VideoCapture(0);
    #Starting to record
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNumber=sampleNumber+1;
        cv2.imwrite(parentDir+"/dataSet/User"+str(ID)+"."+str(sampleNumber)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100)
    cv2.imshow('Face', img)
    #Exitng the programm
    if(sampleNumber>20):
        break;
cam.release()
cv2.destroyAllWindows()
