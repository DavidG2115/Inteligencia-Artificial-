# Proyecto 2

## Características

Este repositorio contiene un juego hecho en Python/Pygame donde un mono esquiva balas en modo manual o automático usando dos modelos de IA:  
- **Red Neuronal (MLPClassifier)**  
- **Arbol de Decisión (DecisionTreeClassifier)**  
- **K-Vecinos (KNeighborsClassifier)**  


# Controles y menú

Al iniciar el juego, aparece un menú con las siguientes opciones:

- **N** → Entrenar red neuronal y entrar en modo automático
- **A** → Entrenar árbol de decisión y entrar en modo automático
- **K** → Entrenar K-vecinos y entrar en modo automático  
- **M** → Modo manual (el jugador controla al mono)
- **R** → Reiniciar dataset y modelos (borra `datos_entrenamiento.csv`)
- **Q** → Salir del juego

## Controles en modo manual

- **⬆️** → Saltar
- **◀️** → Moverse a la izquierda
- **▶️** → Moverse a la derecha
- **P** → Pausar y volver al menú

## Recolección de datos (CSV)

Cada frame en modo manual, se guarda una fila en `datos_entrenamiento.csv` con:

- **velocidad_bala**: velocidad horizontal actual de la bala principal.
- **distancia_bala**: distancia horizontal entre mono y bala principal.
- **distancia_bala2**: distancia vertical entre mono y segunda bala.
- **accion**:
  - `0` → el mono no hizo nada (se centró)
  - `1` → salto
  - `2` → moverse a la izquierda
  - `3` → moverse a la derecha

