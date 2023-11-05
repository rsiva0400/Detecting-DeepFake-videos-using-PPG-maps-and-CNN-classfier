import cv2, numpy as np
import dlib
import tensorflow as tf
import matplotlib.pyplot as plt

face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor('shape_predictor_68_face_landmarks_GTX.dat')

def FacialLandmarks(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(grey)
    try:
        landmarks = dlib_facelandmark(grey, faces[0])
        return landmarks
    except IndexError:
        return []


def roiRegion(img,x,y):
    roi_w = 40
    roi_l = 20
    roi_x = x - roi_w//2
    roi_y = y - roi_l//2
    roi = img[roi_y:roi_y+roi_l, roi_x:roi_x+roi_w, 1]
    return roi

def Flattening(roi):
    tmp = tf.reshape(roi,shape=(1,20,40,1))
    tmp = tf.dtypes.cast(tmp, tf.float32)
    max_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(5,5),strides=(5,5), padding='valid')
    maxpool =  max_pool_2d(tmp)
    return tf.reshape(maxpool, shape=(1,32))

def ppgmap(filename):
    video_path = f'Dataset/Deepfakes/c23/videos/{filename}.mp4'
    cap = cv2.VideoCapture(video_path)
    ppgMap = np.zeros(shape=(128,32))
    frames = []
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    count = 0
    segmentSize = len(frames)//128 if len(frames)//128 != 0 else 1
    for frameInd in range(0,len(frames),segmentSize):
        
        landmarks = FacialLandmarks(frames[frameInd])
        if landmarks != []:
            roi = roiRegion(frames[frameInd],landmarks.part(30).x, landmarks.part(30).y)
            k = Flattening(roi)
            ppgMap[count] = k.numpy()
        count+=1
        if count == 128:
            break
    cv2.imwrite(f"FAke/{filename}.png",ppgMap)
    print(f"{filename} is completed")


import os
files = os.listdir('Dataset/Deepfakes/c23/videos')[0:1000]
# print(files)

for i in files:
    ppgmap(i.split('.')[0])
