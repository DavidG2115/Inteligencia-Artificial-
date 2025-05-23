import pygame
import random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_text
import pandas as pd
import joblib
import os

# Inicializar Pygame
pygame.init()
 
# Definir el tipo de modelo a usar nn = "red neuronal"  dt = "arbol de decision" knn = "k-vecinos"
modelo_tipo = "nn"
modelo = None  # Modelo de red neuronal
modelo_knn = None  # Modelo de k-vecinos
modelo_arbol = None  # Modelo de árbol de decisión


# Dimensiones de la pantalla
w, h = 1000, 600
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
jugador = pygame.Rect(180, h - 120, 32, 48)
bala = pygame.Rect(w - 50, h - 110, 16, 16)
nave = pygame.Rect(w - 100, h - 150, 64, 64)
menu_rect = pygame.Rect(w // 2 - 155, h // 2 - 90, 270, 180)  # Tamaño del menú

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0
velocidad_jugador = 5  # Puedes ajustar la velocidad si quieres

# Variables para la bala
velocidad_bala = -10  # Velocidad de la bala hacia la izquierda
bala_disparada = False

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

# Función para guardar datos de entrenamiento
def guardar_datos():
    global jugador, bala, velocidad_bala, salto
    if modo_auto:  # Solo guardamos en modo manual
        return
    distancia = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0
    datos_modelo.append((velocidad_bala, distancia, salto_hecho))

        
# Función para crear/entrenar el modelo
def crear_entrenar_modelo():
    try:
        df = pd.read_csv('datos_entrenamiento.csv')
        if df.empty or len(df) < 5:
            print("No hay suficientes datos para entrenar la red neuronal.")
            return None
    except:
        print("No se pudo cargar el archivo de datos.")
        return None

    # Eliminar duplicados exactos
    df = df.drop_duplicates()

    # Eliminar filas con valores imposibles
    df = df[(df['velocidad'] <= -1) & (df['velocidad'] >= -20)]  # por ejemplo
    df = df[(df['distancia'] >= 0) & (df['distancia'] <= 1000)]  # lógica básica
    df = df[df['salto'].isin([0, 1])]  # asegurarse que salto sea 0 o 1

    # Eliminar filas faltantes o mal formadas
    df = df.dropna()


    X = df[['velocidad', 'distancia']]
    y = df['salto']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    modelo = MLPClassifier(hidden_layer_sizes=(16, 8,), max_iter=5000)
    modelo.fit(X_train, y_train)

    acc = modelo.score(X_test, y_test)
    print(f"Modelo de red neuronal entrenado con precisión: {acc * 100:.2f}%")

    joblib.dump(modelo, 'modelo_saltos.joblib')
    return modelo
    
def crear_arbol_decision():
    try:
        df = pd.read_csv('datos_entrenamiento.csv')
        if df.empty or len(df) < 10:
            raise ValueError("Archivo vacío o con pocos datos.")
    except:
        print("Modelo No entrenado")
        return None

    X = df[['velocidad', 'distancia']]
    y = df['salto']

    modelo = DecisionTreeClassifier()
    modelo.fit(X, y)
    
    # print(export_text(modelo, feature_names=["velocidad", "distancia"]))

    acc = modelo.score(X, y)  # árboles no generalizan tanto, precisión sobre el mismo dataset
    print(f"Árbol de decisión entrenado con precisión: {acc * 100:.2f}%")

    joblib.dump(modelo, 'modelo_arbol.joblib')
    return modelo
    
    
def crear_modelo_knn():
    try:
        df = pd.read_csv('datos_entrenamiento.csv')
        if df.empty or len(df) < 10:
            raise ValueError("Datos insuficientes")
    except:
        print("Modelo K-Vecinos no entrenado")
        return None

    X = df[['velocidad', 'distancia']]
    y = df['salto']

    n_max = len(X)
    n_vecinos = 7 if n_max >= 7 else n_max

    modelo = KNeighborsClassifier(n_neighbors=n_vecinos, weights='distance')
    modelo.fit(X, y)

    acc = modelo.score(X, y)
    print(f"K-Vecinos entrenado con precisión: {acc * 100:.2f}%")

    joblib.dump(modelo, 'modelo_knn.joblib')
    return modelo


# Cargar modelos
try:
    modelo = joblib.load('modelo_saltos.joblib')
except:
    modelo = crear_entrenar_modelo()

try:
    modelo_arbol = joblib.load('modelo_arbol.joblib')
except:
    modelo_arbol = crear_arbol_decision()

try:
    modelo_knn = joblib.load('modelo_knn.joblib')
except:
    modelo_knn = crear_modelo_knn()

# Verificar si el dataset tiene datos suficientes (una sola vez)
try:
    df_check = pd.read_csv('datos_entrenamiento.csv')
    if df_check.empty or len(df_check) < 5:
        datos_validos = False
    else:
        datos_validos = True
except:
    datos_validos = False

    
def borrar_entrenamiento():
    # Vaciar el dataset en lugar de eliminarlo
    with open("datos_entrenamiento.csv", "w") as f:
        f.write("velocidad,distancia,salto\n")  # Solo encabezado
    print("Dataset vaciado.")

    if os.path.exists("modelo_saltos.joblib"):
        os.remove("modelo_saltos.joblib")
        print("Modelo entrenado eliminado.")
    if os.path.exists("modelo_arbol.joblib"):
        os.remove("modelo_arbol.joblib")
        print("Modelo de árbol de decisión eliminado.")
    if os.path.exists("modelo_knn.joblib"):
        os.remove("modelo_knn.joblib")
        print("Modelo de K-Vecinos eliminado.")

    # Reiniciar el modelo a su estado inicial
    global modelo, modelo_arbol, modelo_knn
    modelo = crear_entrenar_modelo()
    modelo_arbol = crear_arbol_decision()
    modelo_knn = crear_modelo_knn()


# Función para disparar la bala
def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-8, -3)  # Velocidad aleatoria negativa para la bala
        bala_disparada = True

# Función para reiniciar la posición de la bala
def reset_bala():
    global bala, bala_disparada
    bala.x = w - 70  # Reiniciar la posición de la bala
    bala_disparada = False

# Función para manejar el salto
def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura  # Mover al jugador hacia arriba
        salto_altura -= gravedad  # Aplicar gravedad (reduce la velocidad del salto)

        # Si el jugador llega al suelo, detener el salto
        if jugador.y >= h - 120:
            jugador.y = h - 120
            salto = False
            salto_altura = 18  # Restablecer la velocidad de salto
            en_suelo = True

def actualizar_ia():
    
    if not datos_validos:
        return False
    
    if modelo_tipo == "nn" and modelo is None:
        return False
    if modelo_tipo == "dt" and modelo_arbol is None:
        return False
    if modelo_tipo == "knn" and modelo_knn is None:
        return False
    
    distancia = abs(jugador.x - bala.x)
    entrada = pd.DataFrame([[velocidad_bala, distancia]], columns=["velocidad", "distancia"])

    if modelo_tipo == "nn":
        probabilidad_salto = modelo.predict_proba(entrada)[0][1]
        prediccion = 1 if probabilidad_salto > 0.80 else 0
        print(f"Predicción Red Neuronal: {probabilidad_salto:.2f}")
    elif modelo_tipo == "dt":
        probabilidad_salto = modelo_arbol.predict_proba(entrada)[0][1]
        prediccion = 1 if probabilidad_salto > 0.80 else 0
        print(f"Predicción Árbol de Decisión: {probabilidad_salto:.2f}")
    elif modelo_tipo == "knn":
        probabilidad_salto = modelo_knn.predict_proba(entrada)[0][1]
        prediccion = 1 if probabilidad_salto > 0.80 else 0
        print(f"Predicción K-Vecinos: {probabilidad_salto:.2f}")
    else:
        return False

    return prediccion == 1 and en_suelo



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

    # Si la bala sale de la pantalla, reiniciar su posición
    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    # Colisión entre la bala y el jugador
    if jugador.colliderect(bala):
        print("Colisión detectada!")
        reiniciar_juego()  # Terminar el juego y mostrar el menú


# Función para pausar el juego y guardar los datos
def pausa_juego():
    global pausa, modo_auto, modelo, modelo_tipo, modelo_arbol, modelo_knn

    pausa = True
    pantalla.fill(NEGRO)

    # Mostrar estado actual del modo y modelo
    estado_texto = f"Modo: {'Automático' if modo_auto else 'Manual'} | Modelo: {'Red Neuronal' if modelo_tipo == 'nn' else 'Árbol de Decisión'}"
    estado_render = fuente.render(estado_texto, True, BLANCO)
    pantalla.blit(estado_render, (w // 6, h // 3 - 40))

    opciones = [
        "P: Continuar",
        "A: Cambiar a modo Automático",
        "M: Cambiar a modo Manual",
        "N: Usar Red Neuronal",
        "D: Usar Árbol de Decisión",
        "K: Jugar con K-Vecinos",
        "T: Reentrenar Modelos",
        "R: Reiniciar entrenamiento",
        "Q: Salir del juego"
    ]

    for i, texto in enumerate(opciones):
        linea = fuente.render(texto, True, BLANCO)
        pantalla.blit(linea, (w // 6, h // 3 + i * 30))

    pygame.display.flip()

    esperando = True
    while esperando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_p:
                    pausa = False
                    esperando = False
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    print("Modo cambiado a Manual.")
                elif evento.key == pygame.K_n:
                    modo_auto = True
                    pausa = False
                    esperando = False
                    modelo_tipo = "nn"
                    print("Modelo cambiado a Red Neuronal.")
                elif evento.key == pygame.K_d:
                    modo_auto = True
                    pausa = False
                    esperando = False
                    modelo_tipo = "dt"
                    print("Modelo cambiado a Árbol de Decisión.")
                elif evento.key == pygame.K_k:
                    modo_auto = True
                    pausa = False
                    esperando = False
                    modelo_tipo = "knn"
                    print("Modelo cambiado a K-Vecinos.")
                elif evento.key == pygame.K_t:
                    modelo = crear_entrenar_modelo()
                    modelo_arbol = crear_arbol_decision()
                    modelo_knn = crear_modelo_knn()
                    print("Modelos reentrenados.")
                elif evento.key == pygame.K_r:
                    borrar_entrenamiento()
                    print("Entrenamiento y modelos reiniciados.")
                elif evento.key == pygame.K_q:
                    print("Juego terminado.")
                    pygame.quit()
                    exit()
                else:
                    print("Tecla no válida. Presiona 'P' para continuar o 'Q' para salir.")
    


# Función para mostrar el menú y seleccionar el modo de juego
def mostrar_menu():
    global menu_activo, modo_auto, modelo, modelo_tipo, modelo_arbol, modelo_knn

    pantalla.fill(NEGRO)
    texto = fuente.render("MENÚ PRINCIPAL", True, BLANCO)
    pantalla.blit(texto, (w // 3, h // 3 - 40))

    opciones = [
        "N: Jugar con Red Neuronal (Automatico)",
        "D: Jugar con Arbol de Decisión (Automatico)",
        "K: Jugar con K-Vecinos (Automatico)",
        "M: Jugar en Modo Manual",
        "T: Reentrenar Modelos",
        "R: Reiniciar entrenamiento",
        "Q: Salir del juego"
    ]

    for i, linea in enumerate(opciones):
        render = fuente.render(linea, True, BLANCO)
        pantalla.blit(render, (w // 6, h // 3 + i * 30))

    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_n:
                    modo_auto = True
                    modelo_tipo = "nn"
                    menu_activo = False
                elif evento.key == pygame.K_d:
                    modo_auto = True
                    modelo_tipo = "dt"
                    menu_activo = False
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                elif evento.key == pygame.K_k:
                    modo_auto = True
                    modelo_tipo = "knn"
                    menu_activo = False
                elif evento.key == pygame.K_t:
                    modelo = crear_entrenar_modelo()
                    modelo_arbol = crear_arbol_decision()
                    modelo_knn = crear_modelo_knn()
                    print("Modelos reentrenados.")
                elif evento.key == pygame.K_r:
                    borrar_entrenamiento()
                    print("Entrenamiento reiniciado.")
                elif evento.key == pygame.K_q:
                    pygame.quit()
                    exit()



# Función para reiniciar el juego tras la colisión
def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo, modelo, modelo_arbol, modelo_knn, datos_modelo
    
    # Si fue modo manual, guardar y entrenar con nuevos datos
    if not modo_auto and datos_modelo:
        print("Guardando datos...")
        with open('datos_entrenamiento.csv', 'a') as f:
            for v, d, s in datos_modelo:
                f.write(f"{v},{d},{s}\n")
        modelo = crear_entrenar_modelo()
        modelo_arbol = crear_arbol_decision()
        modelo_knn = crear_modelo_knn()
        datos_modelo.clear()
    
    menu_activo = True  # Activar de nuevo el menú
    jugador.x, jugador.y = 180, h - 120  # Reiniciar posición del jugador
    bala.x = w - 50  # Reiniciar posición de la bala
    nave.x, nave.y = w - 100, h - 120  # Reiniciar posición de la nave
    bala_disparada = False
    salto = False
    en_suelo = True
    mostrar_menu() 
    

def main():
    global salto, en_suelo, bala_disparada

    reloj = pygame.time.Clock()
    mostrar_menu()  # Mostrar el menú al inicio
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_UP and en_suelo and not pausa:  # Detectar la tecla espacio para saltar
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:  # Presiona 'p' para pausar el juego
                    pausa_juego()
                if evento.key == pygame.K_LEFT:
                    jugador.x -= velocidad_jugador
                    if jugador.x < 0:
                        jugador.x = 0
                if evento.key == pygame.K_RIGHT:
                    jugador.x += velocidad_jugador
                    if jugador.x > 180:
                        jugador.x = 180
                if evento.key == pygame.K_q:  # Presiona 'q' para terminar el juego
                    pygame.quit()    
                    exit()

        if not pausa:
    # Ejecutar salto si está activo, en cualquier modo
            if salto:
                manejar_salto()

            # Solo guardar datos si es modo manual
            if not modo_auto:
                guardar_datos()

            if not bala_disparada:
                disparar_bala()
            update()

            if modo_auto and actualizar_ia():
                salto = True
                en_suelo = False


        # Actualizar la pantalla
        pygame.display.flip()
        reloj.tick(60)  # Limitar el juego a 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
