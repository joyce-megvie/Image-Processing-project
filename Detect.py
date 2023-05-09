import dlib 
import numpy as np

#loading the face mask detector and landmark predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predict')

#load the two input images
img1 = dlib.load_rgb_image('img1.jpg')
img2 = dlib.load_rgb_image('img2.jpg')

#detect faces in images
face1 = detector(img1)[0]
face2 = detector(img2)[0]

#extract facial landmarks
landmarks1 = predictor(img1, face1)
landmarks2 = predictor(img2, face2)

#compute euclidean distances btween corresponding distances
distances = np.linalg.norm(landmarks1 - landmarks2, axis = 1)

#compute the mean distance as a measure of similarity between the faces
similarity = np.mean(distances)