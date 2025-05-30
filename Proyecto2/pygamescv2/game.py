import pygame
import random
import csv
import os

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Inicializar Pygame
pygame.init()

modelo_nn = None
modelo_arbol = None
scaler_nn = None

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# Variables del jugador, bala, nave, fondo, etc.
jugador = None
bala = None
fondo = None
nave = None
menu = None

# Variables de salto
salto = False
salto_altura = 15  # Velocidad inicial de salto
gravedad = 1
en_suelo = True

# Variables de pausa y menú
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False  # Indica si el modo de juego es automático

# Lista para guardar los datos de velocidad, distancia y salto (target)
datos_modelo = []

# Cargar las imágenes
jugador_frames = [
    pygame.image.load('assets/sprites/mono_frame_1.png'),
    pygame.image.load('assets/sprites/mono_frame_2.png'),
    pygame.image.load('assets/sprites/mono_frame_3.png'),
    pygame.image.load('assets/sprites/mono_frame_4.png')
]

bala_img = pygame.image.load('assets/sprites/purple_ball.png')
fondo_img = pygame.image.load('assets/game/fondo2.png')
nave_img = pygame.image.load('assets/game/ufo.png')
menu_img = pygame.image.load('assets/game/menu.png')

# Escalar la imagen de fondo para que coincida con el tamaño de la pantalla
fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear el rectángulo del jugador y de la bala
jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
bala2 = pygame.Rect( 50, -16, 16, 16)  # Segunda bala para el modo automático
nave = pygame.Rect(w - 100, h - 100, 64, 64)
nave2 = pygame.Rect(jugador.x, -64, 64, 64)  # Segunda nave para el modo automático

menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)  # Tamaño del menú

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0

# Variables para la bala
velocidad_bala = -10  # Velocidad de la bala hacia la izquierda
bala_disparada = False

velocidad_bala2 = 5  
bala2_disparada = False  

velocidad_jugador = 5  # Velocidad del jugador

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

def entrenar_red_neuronal():
    global datos_modelo, modelo_nn, scaler_nn

    datos = []
    try:
        with open("datos_entrenamiento.csv", mode="r", newline="") as archivo:
            lector = csv.reader(archivo)
            next(lector)  # Saltar encabezados
            for fila in lector:
                datos.append([float(fila[0]), float(fila[1]), float(fila[2]), int(fila[3])])
    except FileNotFoundError:
        return False
    

    if not datos:
        return False

    datos = np.array(datos, dtype=float)
    X = datos[:, :3]
    y = datos[:, 3].astype(int)

    scaler_nn = MinMaxScaler()
    X_norm = scaler_nn.fit_transform(X)

    modelo_nn = MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=4000,
        random_state=42,
        activation='relu'
    )
    print("Entrenando red neuronal…")
    modelo_nn.fit(X_norm, y)
    print("Entrenamiento completado.")
    return True


def entrenar_arbol_decision():
    global datos_modelo, modelo_arbol

    datos = []
    try:
        with open("datos_entrenamiento.csv", mode="r", newline="") as archivo:
            lector = csv.reader(archivo)
            next(lector)  # Saltar encabezados
            for fila in lector:
                datos.append([float(fila[0]), float(fila[1]), float(fila[2]), int(fila[3])])
    except FileNotFoundError:
        return False

    if not datos:
        return False

    datos = np.array(datos, dtype=float)
    X = datos[:, :3]
    y = datos[:, 3].astype(int)


    modelo_arbol = DecisionTreeClassifier(max_depth=5)
    modelo_arbol.fit(X, y)
    print("Entrenamiento completado.")
    return True

def logica_auto(accion):
    global salto, en_suelo
    if accion == 1 and en_suelo:
        salto = True
        en_suelo = False
    if accion == 2:
        jugador.x = max(0, jugador.x - velocidad_jugador)
    if accion == 3:
        jugador.x = min(100, jugador.x + velocidad_jugador)
    if accion == 0:
        if jugador.x < 50:
            jugador.x += velocidad_jugador
        elif jugador.x > 50:
            jugador.x -= velocidad_jugador
    if salto:
        manejar_salto()

# Función para disparar la bala
def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-10, -6)
        bala_disparada = True

def disparar_bala2():
    global bala2_disparada, velocidad_bala2
    if not bala2_disparada:
        bala2_disparada = True



# Función para reiniciar la posición de la bala
def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50  # Reiniciar la posición de la bala
    bala_disparada = False

def reset_bala2():
    global bala2, bala2_disparada
    bala2.x = 50  # Reiniciar la posición de la segunda bala
    bala2.y = -16  # Reiniciar la posición vertical de la segunda bala
    bala2_disparada = False

# Función para manejar el salto
def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura  # Mover al jugador hacia arriba
        salto_altura -= gravedad  # Aplicar gravedad (reduce la velocidad del salto)

        # Si el jugador llega al suelo, detener el salto
        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 15  # Restablecer la velocidad de salto
            en_suelo = True

# Función para actualizar el juego
def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2

    # Mover el fondo
    fondo_x1 -= 1
    fondo_x2 -= 1

    # Si el primer fondo sale de la pantalla, lo movemos detrás del segundo
    if fondo_x1 <= -w:
        fondo_x1 = w

    # Si el segundo fondo sale de la pantalla, lo movemos detrás del primero
    if fondo_x2 <= -w:
        fondo_x2 = w

    # Dibujar los fondos
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0

    # Dibujar el jugador con la animación
    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))

    # Dibujar la nave
    pantalla.blit(nave_img, (nave.x, nave.y))

    # Mover y dibujar la bala
    if bala_disparada:
        bala.x += velocidad_bala
    
    if bala2_disparada:
        bala2.y += velocidad_bala2

    # Si la bala sale de la pantalla, reiniciar su posición
    if bala.x < 0:
        reset_bala()

    if bala2.y > h:
        reset_bala2()


    # Dibujar la bala
    pantalla.blit(bala_img, (bala.x, bala.y))
    pantalla.blit(bala_img, (bala2.x, bala2.y))    


    # Colisión entre la bala y el jugador
    if jugador.colliderect(bala) or jugador.colliderect(bala2):
        print("Colisión detectada!")
        reiniciar_juego()  # Terminar el juego y mostrar el menú

# Función para guardar datos del modelo en modo manual
def guardar_datos():
    global jugador, bala, bala2, velocidad_bala, salto
    v = velocidad_bala
    d1 = abs(jugador.x - bala.x)
    d2 = abs(jugador.y - bala2.y)

    accion = 0

    if salto:
        accion = 1
    else:
        if jugador.x < 50 - jugador.width//2:
            accion = 2
        if jugador.x > 50 + jugador.width//2:
            accion = 3

    fila = [v, d1, d2, accion]
    datos_modelo.append(fila)

    # Guardar en CSV cada vez que se llama la función
    with open("datos_entrenamiento.csv", mode="a", newline="") as archivo:
        escritor = csv.writer(archivo)
        # Si el archivo está vacío, escribe los encabezados
        if archivo.tell() == 0:
            escritor.writerow(['velocidad_bala', 'distancia_bala', 'distancia_bala2', 'accion'])
        escritor.writerow(fila)
        

# Función para pausar el juego y guardar los datos
def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado. Datos registrados hasta ahora:")
    else:
        print("Juego reanudado.")

def reiniciar_dataset():
    global datos_modelo, modelo_nn, scaler_nn
    datos_modelo.clear()
    modelo_nn = None
    scaler_nn = None
    if os.path.exists("datos_entrenamiento.csv"):
        os.remove("datos_entrenamiento.csv")
    print("¡Dataset y modelo reiniciados!")

# Función para mostrar el menú y seleccionar el modo de juego
def mostrar_menu():
    global menu_activo, modo_auto, pausa, modelo_nn, modelo_arbol
    pantalla.fill(NEGRO)
    texto = fuente.render("N: Red Neuronal | A: Árbol | M: Manual | R: Reiniciar | Q: Salir", True, BLANCO)
    pantalla.blit(texto, (w // 8, h // 2))
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_n:
                    if entrenar_red_neuronal():
                        modo_auto = "nn"
                        pausa = False
                        menu_activo = False
                    else: 
                        modo_auto = "nn"
                        pausa = False
                        menu_activo = False
                elif evento.key == pygame.K_a:
                    if entrenar_arbol_decision():
                        modo_auto = "arbol"
                        pausa = False
                        menu_activo = False
                    else:
                        modo_auto = "arbol"
                        pausa = False
                        menu_activo = False
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                elif evento.key == pygame.K_r:
                    reiniciar_dataset()
                    pantalla.fill(NEGRO)
                    texto = fuente.render("Dataset reiniciado. Elige modo.", True, BLANCO)
                    pantalla.blit(texto, (w // 4, h // 2))
                    pygame.display.flip()
                    pygame.time.wait(1000)
                    mostrar_menu()
                    return
                elif evento.key == pygame.K_q:
                    pygame.quit()
                    exit()

# Función para reiniciar el juego tras la colisión
def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo
    menu_activo = True  # Activar de nuevo el menú
    jugador.x, jugador.y = 50, h - 100  # Reiniciar posición del jugador
    bala.x = w - 50  # Reiniciar posición de la bala
    bala2.y = 50
    nave.x, nave.y = w - 100, h - 100  # Reiniciar posición de la nave
    bala_disparada = False
    bala2_disparada = False
    salto = False
    en_suelo = True
    # Mostrar los datos recopilados hasta el momento
    # print("Datos recopilados para el modelo: ", datos_modelo)
    mostrar_menu()  # Mostrar el menú de nuevo para seleccionar modo

def main():
    global salto, en_suelo, bala_disparada, modo_auto, pausa

    reloj = pygame.time.Clock()
    mostrar_menu()  # Mostrar el menú al inicio
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_UP and en_suelo and not pausa:  # Saltar
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:
                    pausa_juego()
                    mostrar_menu()
                if evento.key == pygame.K_q:  # Salir
                    pygame.quit()
                    exit()

        if not pausa:
            if not modo_auto:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    jugador.x = max(0, jugador.x - velocidad_jugador)
                elif keys[pygame.K_RIGHT]:
                    jugador.x = min(100, jugador.x + velocidad_jugador)  # <-- Límite derecho en 400
                else:
                    if jugador.x < 50:
                        jugador.x = min(50, jugador.x + velocidad_jugador)
                    elif jugador.x > 50:
                        jugador.x = max(50, jugador.x - velocidad_jugador)

                if salto:
                    manejar_salto()
                guardar_datos()
            else:
                # lógica modo automático
                if modo_auto == "nn":
                    if modelo_nn is not None and scaler_nn is not None:
                        v = velocidad_bala
                        d1 = abs(jugador.x - bala.x)
                        d2 = abs(jugador.y - bala2.y)
                        x_input = np.array([[v, d1, d2]])
                        x_input_norm = scaler_nn.transform(x_input)
                        accion = modelo_nn.predict(x_input_norm)[0]
                        proba = modelo_nn.predict_proba(x_input_norm)[0]
                        logica_auto(accion)
                        print(f"NN Acción predicha: {accion}, Probabilidades: {proba}")
                elif modo_auto == "arbol":
                    if modelo_arbol is not None:
                        v = velocidad_bala
                        d1 = abs(jugador.x - bala.x)
                        d2 = abs(jugador.y - bala2.y)
                        x_input = np.array([[v, d1, d2]])
                        accion = modelo_arbol.predict(x_input)[0]
                        logica_auto(accion)
                        print(f"Árbol Acción predicha: {accion}")
                else:
                    # No hay modelo entrenado: el mono no hace nada
                    pass
                

            # Actualizar el juego
            if not bala_disparada:
                disparar_bala()
            elif not bala2_disparada:
                disparar_bala2()
            update()

        pygame.display.flip()
        reloj.tick(30)  # Limitar el juego a 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()