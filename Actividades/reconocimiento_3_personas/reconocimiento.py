import cv2 as cv
import numpy as np
import os


# Cargar modelo
model_name = 'modelo_facial_eigen.xml'


recognizer = cv.face.EigenFaceRecognizer_create()
recognizer.read(model_name)
label_dict = np.load('label_dict.npy', allow_pickle=True).item()

# Iniciar cámara
cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv.resize(face_roi, (100, 100))
        
        label, confidence = recognizer.predict(face_roi)
        
        if confidence < 3000:  # Umbral ajustable
            name = label_dict.get(label, "Desconocido")
            color = (0, 255, 0)
        else:
            name = "Desconocido"
            color = (0, 0, 255)
        
        cv.putText(frame, f"{name} ({confidence:.1f})", (x, y-10), 
                 cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    cv.imshow('Reconocimiento Facial', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()