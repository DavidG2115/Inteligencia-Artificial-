import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,        # Número máximo de rostros a detectar
    refine_landmarks=True,  # Incluye landmarks de iris (468 → 478 puntos)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Configuración de dibujo
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Iniciar cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    # Convertir a RGB y procesar
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    
    # Dibujar landmarks
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar conexiones (malla completa)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,  # Malla principal
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )
            
            # Opcional: Dibujar contornos faciales (líneas gruesas)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    thickness=1, 
                    color=(0, 255, 255))  # Amarillo para contornos
            )
    
    # Mostrar FPS
    cv2.putText(image, "Face Mesh", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))  # Espejo
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()