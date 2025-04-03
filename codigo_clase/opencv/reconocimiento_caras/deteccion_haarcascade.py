import numpy as np
import cv2 as cv

# Cargar el clasificador de rostros
rostro = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Iniciar la captura de video desde la cámara
#cap = cv.VideoCapture(0)  # Usar la cámara
cap = cv.VideoCapture("C:/Users/garcd/Downloads/Ana de armas.mp4")  # Cambiar para usar un archivo de video

i = 0  # Contador para nombrar las imágenes guardadas

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    rostros = rostro.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in rostros:
        # Dibujar un rectángulo alrededor del rostro en el frame original
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # Recortar la región del rostro
        rostro_recortado = frame[y:y + h, x:x + w]

        # Redimensionar el rostro recortado a 100x100 píxeles
        rostro_recortado = cv.resize(rostro_recortado, (100, 100), interpolation=cv.INTER_AREA)

        # Mostrar el rostro recortado en una ventana
        cv.imshow('Rostro Recortado', rostro_recortado)

        # Guardar el rostro recortado cada 5 frames
        if i % 5 == 0:
            cv.imwrite(f'C:/Users/garcd/OneDrive/Desktop/IA/codigo_clase/opencv/recortes/ana/{i}.jpg', rostro_recortado)

    # Mostrar el frame completo con los rostros detectados
    cv.imshow('Video', frame)

    i += 1

    # Salir si se presiona la tecla 'ESC'
    if cv.waitKey(1) & 0xFF == 27:
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv.destroyAllWindows()