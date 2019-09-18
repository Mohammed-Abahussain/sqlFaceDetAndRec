import os
import cv2
import numpy as np
parentDir=os.path.abspath(os.path.join(os.getcwd(), os.pardir))
faceDetect=cv2.CascadeClassifier(parentDir+'/haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
recognizer=cv2.face.LBPHFaceRecognizer_create()
parentDir=os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(parentDir)
Yaml=os.path.abspath(os.path.join(parentDir, 'DataSetter/TrainedData.yml'))
recognizer.read(Yaml)

id=0
redRGB=(0,0,255)
font= cv2.FONT_HERSHEY_SIMPLEX

cam=cv2.VideoCapture(0);
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        cv2.putText(img, str(id), (x,y+h),font,4,redRGB) 

    cv2.imshow('Face', img)
    #Exitng the programm
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
