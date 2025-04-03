

#Evaluacion Redes Neuronales Mediapipe

Nombre: David Garcia Aburto Calificacion

##Modelar una red neuronal que pueda identificar emociones a travez de los valores obtenidos de los landmarks que genera mediapipe.

- Definir el tipo de red neuronal y describir cada una de sus partes

    - El tipo de red neuronal es una Red neuronal Full connected, las capas que va a tener son:
    - Capa de entrada: Recibe los valores de los landmarks generados con Mediapipe
    - Capas ocultas: Procesan las caracteristicas de los datos
    - Capa de salida: genera las probabilidades de las emociones detectadas

- Definir los patrones a utilizar
Los patrones seran los valores que nos va a estar dando los landmarks generador por Mediapipe
    - Landmakrs faciales: Se van a tener en cuenta las coordenadas de los puntos clave, ya sea en los ojos, cejas, labios, mandibula, pomulos
    - Distancias y angulos: Se tendra en cuenta las distancias que hay en puntos especificos como la ditancia entre las esquinas de los labios, la distancia entre el labio superior e inferior para saber si abrio la boca, la distancia entre el parpado superior e inferior para saber si parpadeo la persona, tambien en los labios se pudiera tomar el angulo que hacen los labios para saber si esta triste una persona, la distancia entre las cejas para saber si esta enojado
    - Variaciones temporales: Para saber que una persona esta viva, se tomaran en cuenta los pequeños movimientos comparando los fotogramas
    

- Definir funcion de activacion es necesaria para este problema


- Definir numero maximo de entradas
Mediapipe nos genera una estimacion de 478 puntos de referencia tridimencional, por lo que tenenos que tener en cuentas las 3 dimenciones el resultado seria: 478 x 3 = 1334

¿Que valores a la salia de la red se podria esperar?

La salida sera un vector de probabilidades donde cada posicion representa la probabilidad de que sea una emocion por ejemplo:
[0.1, 0.7, 0.2] lo cual representaria 10% tristeza, 70% felicidad, 20% sorpresa.


¿Cuales son los valores maximos que puede tener el bias?

no se tiene un valor fijo ya que se ajusta durante el entrenamiento pero se empezara con un valor 0.01