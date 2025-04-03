# Actividad: Reconocimiento Facial con OpenCV y Haarcascade

## Descripción

Esta actividad consiste en desarrollar un sistema de **reconocimiento facial** que se divide en tres etapas principales:

1. **Generacion de dataset** desde un video o cámara.
2. **Entrenamiento de un modelo facial** con EigenFaces.
3. **Reconocimiento en tiempo real** usando el modelo entrenado.

---

## 1. Generación del Dataset

Se utiliza OpenCV para detectar rostros en un video o cámara, recortarlos, redimensionarlos a 100x100 píxeles y guardarlos como imagenes. Esto se realiza en el archivo deteccion.py

## 2. Entrenamiento del Modelo
Se entrena un modelo usando cv.face.EigenFaceRecognizer_create() a partir del dataset creado previamente.

Cada carpeta representa una persona distinta.

Las imagenes son convertidas a escala de grises y redimensionadas.

Se genera un modelo .xml y un diccionario de etiquetas .npy.

Archivos generados:
modelo_facial_eigen.xml -> Modelo entrenado

label_dict.npy -> Diccionario de etiquetas

## 3. Reconocimiento Facial en Tiempo Real

Se carga el modelo entrenado y se utiliza la camara en tiempo real para detectar y reconocer rostros si la prediccion tiene un nivel de confianza aceptable, se muestra el nombre de la persona de lo contrario, se muestra como "Desconocido".
Se utilizo el algoritmo: 
- **EigenFaceRecognizer** de OpenCV

Para esto tenemos los parametros definidos del umbral en <3000 para que sea reconocido, de lo contrario se marcara como desconocido
### Umbral:
```python
if confidence < 3000:
    # Reconocido
else:
    # Desconocido


