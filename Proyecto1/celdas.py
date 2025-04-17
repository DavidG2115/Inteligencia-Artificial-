import pygame, math
from queue import PriorityQueue

# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos")


# Pesos configurables
PESO_RECTO = 1
PESO_DIAGONAL = 1.414  # Aproximación de √2


# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AZUL = (0, 0, 255)

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.y = col * ancho
        self.x = fila * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))
        
    def hacer_abierto(self):
        self.color = VERDE
    
    def hacer_cerrado(self):
        self.color = ROJO
        
    def hacer_camino(self):
        self.color = AZUL
        
    def actualizar_vecinos(self, grid):
        self.vecinos = []
        filas = self.total_filas

        direcciones = [
            (-1,  0, PESO_RECTO),  # arriba
            ( 1,  0, PESO_RECTO),  # abajo
            ( 0, -1, PESO_RECTO),  # izquierda
            ( 0,  1, PESO_RECTO),  # derecha
            (-1, -1, PESO_DIAGONAL),  # arriba izquierda
            (-1,  1, PESO_DIAGONAL),  # arriba derecha
            ( 1, -1, PESO_DIAGONAL),  # abajo izquierda
            ( 1,  1, PESO_DIAGONAL),  # abajo derecha
        ]

        for dx, dy, peso in direcciones:
            nueva_fila = self.fila + dx
            nueva_col = self.col + dy

            if 0 <= nueva_fila < filas and 0 <= nueva_col < filas:
                vecino = grid[nueva_fila][nueva_col]
                if not vecino.es_pared():
                    self.vecinos.append((vecino, peso))  # Ahora incluye el peso


def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def heuristica(a, b):
    x1, y1 = a.get_pos()
    x2, y2 = b.get_pos()
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return PESO_RECTO * (dx + dy) + (PESO_DIAGONAL - 2 * PESO_RECTO) * min(dx, dy)



def reconstruir_camino(came_from, actual, dibujar):
    while actual in came_from:
        actual = came_from[actual]
        if not actual.es_inicio():
            actual.hacer_camino()
        dibujar()


def a_estrella(dibujar, grid, inicio, fin):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, inicio))

    came_from = {}

    g_score = {n: float("inf") for fila in grid for n in fila}
    g_score[inicio] = 0

    f_score = {n: float("inf") for fila in grid for n in fila}
    f_score[inicio] = heuristica(inicio, fin)

    open_hash = {inicio}

    while not open_set.empty():
        # permite cerrar la ventana mientras corre
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return False

        actual = open_set.get()[2]
        open_hash.remove(actual)

        if actual == fin:
            reconstruir_camino(came_from, fin, dibujar)
            fin.hacer_fin(); inicio.hacer_inicio()
            return True

        for vecino, peso in actual.vecinos:
            temp_g = g_score[actual] + peso
            if temp_g < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = temp_g
                f_score[vecino] = temp_g + heuristica(vecino, fin)
                if vecino not in open_hash:
                    count += 1
                    open_set.put((f_score[vecino], count, vecino))
                    open_hash.add(vecino)
                    vecino.hacer_abierto()

        dibujar()
        if actual != inicio:
            actual.hacer_cerrado()

    return False        # sin ruta

def main(ventana, ancho):
    pygame.init()
    FILAS = 20
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()

                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()

                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None
            
            # Ejecutar algoritmo A* al presionar la tecla "espacio"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)
                    a_estrella(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)
                    
                # --- 3‑A: limpiar con C ---
                if event.key==pygame.K_c:
                    grid = crear_grid(FILAS, ancho); inicio=fin=None
                    
                

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)