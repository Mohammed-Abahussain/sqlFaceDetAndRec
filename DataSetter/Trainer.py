import os
import cv2
import numpy as np 
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create()



path=(os.path.abspath(os.path.join(os.getcwd(), os.pardir))+"/dataSet/Users")
print ("path"+path)

def getImagesAndID(path):
    imagePaths=[os.path.join(path,name)  for name in os.listdir(path) if name !='.DS_Store']
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceIMG=Image.open(imagePath).convert('L');
        faceNP=np.array(faceIMG,'uint8')
        ID=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceNP)
        IDs.append(ID)
        cv2.imshow("training",faceNP)
        cv2.waitKey(10)
    return IDs,faces
Ids,faces=getImagesAndID(path)
recognizer.train(faces, np.array(Ids))
trainee="trainingData.yml"
print(os.path.join(os.getcwd(),trainee))
recognizer.write(os.path.join(os.getcwd(),trainee))
cv2.destroyAllWindows() 
