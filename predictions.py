import numpy as np
import cv2
import pywt

from joblib import load



face_cascade=cv2.CascadeClassifier(r'opencv_haarcascades\haarcascade_frontalface_default.xml')

eye_cascade =cv2.CascadeClassifier(r'opencv_haarcascades\haarcascade_eye.xml')

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color



def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

model = load('soports_celeberty_classification.pkl')

img=cv2.imread(r'testing images\Shaid Afridi\images.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cropped_img=get_cropped_image_if_2_eyes(r'testing images\Shaid Afridi\images.jpg')

scalled_raw_img = cv2.resize(cropped_img, (32, 32))

img_har = w2d(scalled_raw_img,'db1',5)

scalled_img_har = cv2.resize(img_har, (32, 32))

combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))

combined_img_1= np.array(combined_img).reshape(1,4096).astype(float)

print(model.predict(combined_img_1))
