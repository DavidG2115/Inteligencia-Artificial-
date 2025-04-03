import cv2 as cv 
import numpy as np 
import os


dataSet = 'C:/Users/garcd/OneDrive/Desktop/IA/Actividades/reconocimiento_3_personas/recortes'
model_name = 'modelo_facial_eigen.xml'

# Cargar imágenes
faces = []
labels = []
label_dict = {}
current_label = 0

for person_name in os.listdir(dataSet):
    person_path = os.path.join(dataSet, person_name)
    if os.path.isdir(person_path):
        label_dict[current_label] = person_name
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv.resize(img, (100, 100))
                faces.append(img)
                labels.append(current_label)
        current_label += 1

# Entrenar modelo
recognizer = cv.face.EigenFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save(model_name)

# Guardar el mapeo de etiquetas
np.save('label_dict.npy', label_dict)

print(f"Modelo entrenado y guardado como {model_name}")
print(f"Personas reconocibles: {list(label_dict.values())}")