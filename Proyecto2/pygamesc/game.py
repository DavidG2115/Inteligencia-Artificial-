import pygame
import random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_text
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import load_model

import pandas as pd
import joblib
import os
import time
import matplotlib.pyplot as plt

# Inicializar Pygame
pygame.init()
 
# Definir el tipo de modelo a usar nn = "red neuronal"  dt = "arbol de decision" knn = "k-vecinos"
modelo_tipo = "dt"
modelo = None  # Modelo de red neuronal
modelo_knn = None  # Modelo de k-vecinos
modelo_arbol = None  # Modelo de árbol de decisión
scaler_knn = None


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
bala2 = None
fondo = None
nave = None
nave2 = None
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
bala2 = pygame.Rect(180, h -650, 16, 16)

nave = pygame.Rect(w - 100, h - 150, 64, 64)
nave2 = pygame.Rect(130, h - 600, 64, 64)
menu_rect = pygame.Rect(w // 2 - 155, h // 2 - 90, 270, 180)  # Tamaño del menú

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0
velocidad_jugador = 40  # Puedes ajustar la velocidad si quieres
ultima_accion = 0  # 0: quieto, 1: izquierda, 2: derecha, 3: salto

POS_ORIGEN = 180
POS_ESQUIVA = 120


# Variables para la bala
velocidad_bala = -10  # Velocidad de la bala hacia la izquierda
bala_disparada = False

# Variables para la segunda bala
bala2_disparada = False
velocidad_bala2 = 2

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

ultimo_estado = None
ultimo_guardado = 0
esquiva_exitosa = False


try:
    scaler = joblib.load("scaler_contextual.pkl")
except FileNotFoundError:
    scaler = None
    print("Scaler no encontrado. Entrénalo primero con algunos datos.")


# Función para guardar datos de entrenamiento
def guardar_datos():
    global jugador, bala, bala2, velocidad_bala, velocidad_bala2, ultima_accion, en_suelo

    if modo_auto:
        return  # Solo guardar en modo manual

    # === INPUTS ===
    despBala = jugador.x - bala.x
    velocidadBala = velocidad_bala
    despBala2 = jugador.y - bala2.y
    velocidadBala2 = velocidad_bala2

    # === OUTPUTS ===
    estatusAire = 1 if not en_suelo else 0
    estatusSuelo = 1 if en_suelo else 0
    estatusDerecha = 1 if ultima_accion == 2 else 0
    estatusIzquierda = 1 if ultima_accion == 1 else 0


    archivo = "datos_entrenamiento.csv"
    encabezado = "despBala,velocidadBala,despBala2,velocidadBala2,estatusAire,estatusSuelo,estatusDerecha,estatusIzquierda\n"
    escribir_encabezado = not os.path.exists(archivo) or os.stat(archivo).st_size == 0

    with open(archivo, "a") as f:
        if escribir_encabezado:
            f.write(encabezado)
        f.write(f"{despBala},{velocidadBala},{despBala2},{velocidadBala2},{estatusAire},{estatusSuelo},{estatusDerecha},{estatusIzquierda}\n")




# Función para crear/entrenar el modelo
def crear_entrenar_modelo():
    try:
        df = pd.read_csv("datos_entrenamiento.csv")
        df.drop_duplicates(inplace=True)

        if df.empty or len(df) < 5:
            print("No hay suficientes datos para entrenar.")
            return None
    except Exception as e:
        print(f"Error al leer el CSV: {e}")
        return None

    # Entradas y salidas
    X = df[['despBala', 'velocidadBala', 'despBala2', 'velocidadBala2']]
    y = df[['estatusAire', 'estatusSuelo', 'estatusDerecha', 'estatusIzquierda']]

    # Escalar los datos
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler_contextual.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    model = MLPClassifier(hidden_layer_sizes=(64, 64, 64), activation='relu', solver='adam',
                          max_iter=10000, random_state=42, shuffle=True)

    model = MultiOutputClassifier(model, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluar
    score = model.score(X_test, y_test)
    print(f"Precisión global del modelo: {score * 100:.2f}%")

    # Guardar modelo
    joblib.dump(model, "modelo_mlp.joblib")
    return model


    
def crear_arbol_decision():
    try:
        df = pd.read_csv('datos_entrenamiento.csv')
        df.drop_duplicates(inplace=True)

        if df.empty or len(df) < 5:
            print("No hay suficientes datos para entrenar árbol.")
            return None
    except:
        print("Error al cargar el dataset.")
        return None

    # Entradas y salidas
    X = df[['despBala', 'velocidadBala', 'despBala2', 'velocidadBala2']]
    y = df[['estatusAire', 'estatusSuelo', 'estatusDerecha', 'estatusIzquierda']]

    # Entrenar un árbol por cada salida (multi-output)
    modelo_arbol = DecisionTreeClassifier(max_depth=10, random_state=42)
    modelo_arbol.fit(X, y)

    acc = modelo_arbol.score(X, y)
    print(f"Árbol entrenado con precisión: {acc*100:.2f}%")

    joblib.dump(modelo_arbol, 'modelo_arbol.joblib')
    return modelo_arbol
    
    
def crear_modelo_knn():
    try:
        df = pd.read_csv('datos_entrenamiento.csv')
        df.drop_duplicates(inplace=True)

        if df.empty or len(df) < 10:
            raise ValueError("Datos insuficientes")
    except Exception as e:
        print(f"Error al cargar datos para KNN: {e}")
        return None

    X = df[['despBala', 'velocidadBala', 'despBala2', 'velocidadBala2']]
    y = df[['estatusAire', 'estatusSuelo', 'estatusDerecha', 'estatusIzquierda']]

    # Escalado con MinMaxScaler (importante para KNN)
    scaler_knn = MinMaxScaler()
    X_knn = scaler_knn.fit_transform(X)
    joblib.dump(scaler_knn, 'scaler_knn.pkl')

    n_max = len(X)
    n_vecinos = 5 if n_max >= 5 else n_max

    modelo_knn = KNeighborsClassifier(n_neighbors=n_vecinos, weights='distance')
    modelo_knn.fit(X_knn, y)

    acc = modelo_knn.score(X_knn, y)
    print(f"K-Vecinos entrenado con precisión: {acc * 100:.2f}%")

    joblib.dump(modelo_knn, 'modelo_knn.joblib')
    return modelo_knn


# Cargar modelos
try:
    modelo = joblib.load('modelo_mlp.joblib')
except:
    modelo = crear_entrenar_modelo()

try:
    modelo_arbol = joblib.load('modelo_arbol.joblib')
except:
    modelo_arbol = crear_arbol_decision()

try:
    scaler_knn = joblib.load('scaler_knn.pkl')
except:
    scaler_knn = None

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
        f.write("despBala,velocidadBala,despBala2,velocidadBala2,estatusAire,estatusSuelo,estatusDerecha,estatusIzquierda\n")  # Solo encabezado
    print("Dataset vaciado.")
    if os.path.exists("modelo_mlp.joblib"):
        os.remove("modelo_mlp.joblib")
        print("Modelo entrenado eliminado.")
    if os.path.exists("modelo_arbol.joblib"):
        os.remove("modelo_arbol.joblib")
        print("Modelo de árbol de decisión eliminado.")
    if os.path.exists("modelo_knn.joblib"):
        os.remove("modelo_knn.joblib")
        print("Modelo de K-Vecinos eliminado.")
    if os.path.exists("scaler_contextual.pkl"):
        os.remove("scaler_contextual.pkl")
        print("Scaler eliminado.")
    if os.path.exists("scaler_knn.pkl"):
        os.remove("scaler_knn.pkl")
        print("Scaler KNN eliminado.")

    # Reiniciar el modelo a su estado inicial
    global modelo, modelo_arbol, modelo_knn
    modelo = crear_entrenar_modelo()
    modelo_arbol = crear_arbol_decision()
    modelo_knn = crear_modelo_knn()


# Función para disparar la bala
def disparar_balas():
    global bala_disparada, velocidad_bala, bala2_disparada, velocidad_bala2

    if not bala_disparada:
        velocidad_bala = random.randint(-8, -3)
        bala_disparada = True

    if not bala2_disparada:
        bala2.y = -20
        velocidad_bala2 = random.randint(3, 8) 
        bala2_disparada = True


# Función para reiniciar la posición de la bala
def reset_bala1():
    global bala, bala_disparada

    bala.x = w - 70
    bala_disparada = False
    
def reset_bala2():
    global bala2, bala2_disparada
    bala2.x = 180
    bala2_disparada = False


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
            salto_altura = 15  # Restablecer la velocidad de salto
            en_suelo = True

def actualizar_ia():
    global jugador, bala, bala2, velocidad_bala, velocidad_bala2, salto, en_suelo

    if modelo is None or scaler is None or not datos_validos:
        return False, False, False, False

    # Crear entrada como DataFrame para mantener columnas correctas
    entrada = pd.DataFrame([[
        jugador.x - bala.x,
        velocidad_bala,
        jugador.y - bala2.y,
        velocidad_bala2
    ]], columns=["despBala", "velocidadBala", "despBala2", "velocidadBala2"])

    # Escalar entrada
    entrada_scaled = scaler.transform(entrada)

    # Realizar predicción con modelo MultiOutputClassifier
    pred = modelo.predict_proba(entrada_scaled) # pred → [aire, suelo, der, izq]

    aire_prob = pred[0][0][1]    
    suelo_prob = pred[1][0][1]   
    der_prob = pred[2][0][1]     
    izq_prob = pred[3][0][1]     

    # Umbral de decisión
    aire = aire_prob > 0.6
    suelo_pred = suelo_prob > 0.6
    mover_der =  der_prob > 0.6
    mover_izq =  izq_prob > 0.6

    print(f"Pred MLP → Aire: {aire_prob:.2f}, Suelo: {suelo_prob:.2f}, Der: {der_prob:.2f}, Izq: {izq_prob:.2f}")

    # Ejecutar salto si está en el suelo
    if aire and en_suelo:
        salto = True
        en_suelo = False
        manejar_salto()

    # Movimiento lateral
    if mover_izq:
        jugador.x = POS_ESQUIVA
    elif mover_der:
        jugador.x = POS_ORIGEN

    return aire, suelo_pred, mover_izq, mover_der


def actualizar_ia_arbol():
    global jugador, bala, bala2, velocidad_bala, velocidad_bala2, salto, en_suelo

    if not modelo_arbol or not datos_validos:
        return False, False, False, False

    entrada = pd.DataFrame([[ 
        jugador.x - bala.x,
        velocidad_bala,
        jugador.y - bala2.y,
        velocidad_bala2
    ]], columns=["despBala", "velocidadBala", "despBala2", "velocidadBala2"])

    # Obtener las probabilidades
    probas = modelo_arbol.predict_proba(entrada)

    # Cada salida es un array: [proba_clase_0, proba_clase_1]
    aire = probas[0][0][1] > 0.5
    suelo_pred = probas[1][0][1] > 0.5
    mover_der = probas[2][0][1] > 0.5
    mover_izq = probas[3][0][1] > 0.5

    print(f"Pred Arbol Prob → Aire: {probas[0][0][1]:.2f}, Suelo: {probas[1][0][1]:.2f}, Der: {probas[2][0][1]:.2f}, Izq: {probas[3][0][1]:.2f}")

    if aire and en_suelo:
        salto = True
        en_suelo = False
        manejar_salto()

    if mover_izq:
        jugador.x = POS_ESQUIVA
    elif mover_der:
        jugador.x = POS_ORIGEN

    return aire, suelo_pred, mover_izq, mover_der



def actualizar_ia_knn():
    global jugador, bala, bala2, velocidad_bala, velocidad_bala2, salto, en_suelo

    if not modelo_knn or not datos_validos or not scaler_knn:
        return False, False, False, False

    entrada = pd.DataFrame([[  # usando pandas
        jugador.x - bala.x,
        velocidad_bala,
        jugador.y - bala2.y,
        velocidad_bala2
    ]], columns=["despBala", "velocidadBala", "despBala2", "velocidadBala2"])

    entrada_scaled = scaler_knn.transform(entrada)

    probs = modelo_knn.predict_proba(entrada_scaled)

    aire = probs[0][0][1] > 0.8
    suelo_pred = probs[1][0][1] > 0.8
    mover_der = probs[2][0][1] > 0.8
    mover_izq = probs[3][0][1] > 0.8


    print(f"Pred KNN → Aire: {probs[0][0][1]:.2f}, Suelo: {probs[1][0][1]:.2f}, Der: {probs[2][0][1]:.2f}, Izq: {probs[3][0][1]:.2f}")

    if aire and en_suelo:
        salto = True
        en_suelo = False
        manejar_salto()

    if mover_izq:
        jugador.x = POS_ESQUIVA
    elif mover_der:
        jugador.x = POS_ORIGEN

    return aire, suelo_pred, mover_izq, mover_der





# Función para actualizar el juego
def update():
    global bala, velocidad_bala, bala2, velocidad_bala2, bala2_disparada, bala_disparada, current_frame, frame_count, fondo_x1, fondo_x2, salto

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
    pantalla.blit(nave_img, (nave2.x, nave2.y))

    # Mover y dibujar la bala
    if bala_disparada:
        bala.x += velocidad_bala

    # Si la bala sale de la pantalla, reiniciar su posición
    if bala.x < 0 :
        reset_bala1()

    pantalla.blit(bala_img, (bala.x, bala.y))
    
    # Mover bala2
    if bala2_disparada:
        bala2.y += velocidad_bala2

    # Reiniciar si sale de pantalla
    if bala2.y > h :
        reset_bala2()


    pantalla.blit(bala_img, (bala2.x, bala2.y))


    # Colisión con jugador
    if jugador.colliderect(bala2):
        print("Colisión con bala2 detectada!")
        reiniciar_juego()

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
    global menu_activo, jugador, bala, nave, bala_disparada, bala2_disparada, salto, en_suelo, modelo, modelo_arbol, modelo_knn, datos_modelo
    
    
    menu_activo = True  # Activar de nuevo el menú
    jugador.x, jugador.y = 180, h - 120  # Reiniciar posición del jugador
    bala.x = w - 50  # Reiniciar posición de la bala
    bala2.x = 180  # Reiniciar posición de la bala2
    bala2.y = h - 20  # Reiniciar posición de la bala2
    nave.x, nave.y = w - 100, h - 120  # Reiniciar posición de la nave
    nave2.x, nave2.y = 180, h - 600  # Reiniciar posición de la nave2
    bala_disparada = False
    bala2_disparada = False
    salto = False
    en_suelo = True
    mostrar_menu() 
    

def main():
    global salto, en_suelo, bala_disparada, bala2_disparadam, ultima_accion

    reloj = pygame.time.Clock()
    mostrar_menu()  # Mostrar el menú al inicio
    correr = True

    frame_ia = 0  # contador de frames IA
    pred_salto = False

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_UP and en_suelo and not pausa:  # Detectar la tecla espacio para saltar
                    salto = True
                    en_suelo = False
                    ultima_accion = 3
                if evento.key == pygame.K_p:  # Presiona 'p' para pausar el juego
                    pausa_juego()
                if evento.key == pygame.K_LEFT:
                    jugador.x -= velocidad_jugador
                    ultima_accion = 1
                    if jugador.x < 0:
                        jugador.x = 0
                if evento.key == pygame.K_RIGHT:
                    ultima_accion = 2
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

            if not bala_disparada and not bala2_disparada:
                disparar_balas()
            update()

            if modo_auto:
                frame_ia += 1
                if frame_ia % 1 == 0:
                    # Actualizar IA cada 5 frames
                    if modelo_tipo == "nn":
                        pred_salto, pred_suelo, pred_izquierda, pred_der = actualizar_ia()
                    elif modelo_tipo == "dt":
                        pred_salto, pred_suelo, pred_izquierda, pred_der = actualizar_ia_arbol()
                    elif modelo_tipo == "knn":
                        pred_salto, pred_suelo, pred_izquierda, pred_der = actualizar_ia_knn()
                

                if pred_salto and en_suelo and not salto:
                    salto = True
                    en_suelo = False


        # Mostrar FPS en título
        pygame.display.set_caption(f"FPS: {reloj.get_fps():.2f}")
        pygame.display.flip()
        reloj.tick(60)  # límite de 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
