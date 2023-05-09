import os
import cv2
import numpy as np
from flask import Flask, request, jsonify


app = Flask(__name__)

#load face detection and recognition models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer.yml')

#endpoint definition for uploading images

@app.route('/api/upload', methods=['POST'])
def upload():
    #get the uploaded image from the request
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)


    #convert the image to grayscale

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)

    #recognize faces in the image
    results = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)
        if confidence <= 100:
            results.append
    ({'id':id_, 'confidence': "{:.2f}".format(confidence)})

    #return the results as JSON

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug= True)