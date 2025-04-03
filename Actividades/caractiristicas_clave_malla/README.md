# Planeacion: Verificacion de Vida con Face Mesh

## Objetivo

Diseñar un sistema que permita verificar si una persona esta viva al momento del reconocimiento facial, evitando suplantacion mediante fotografias, videos pausados o capturas de pantalla



## Parametros y caracteristicas clave

### Movimiento Natural del Rostro

- El rostro humano no se mantiene completamente estatico
- Incluso aunque una persona este en reposo hay pequeños movimientos ya sea en las cejas, labios, mandibula, cabeza, etc

### Movimiento y parpadeo de ojos

- El parpadeo es un indicador que una persona tiene vida, ya que lo hacemos inconcientemente y cada cierto tiempo asi como el movimeinto de los ojos, si nos acercamos a algo o simplemente por naturaleza observamos lo que hay alrededor, ya sea que vimos algo diferente o solo por curiosidad
- La distancia entre parpados o metricas para detectar cierres y aperturas de los ojos puede ser un indicador clave que nos ayuda a detectar si es una persona viva
- Si los ojos permanecen abiertos por muchos segundos podria tratarse de una imagen o video

### Expresiones faciales y emociones

- Las emociones provocan cambios visibles en el rostro: principalmente en cejas, ojos y labios.
- Se analizaran distancias y angulos tal vez en la boca para poder detectar si es una sonrisa lo que normalmente hace una persona es hacer como una curva con la boca o incluso abrir la boca y mostrar los dientes para sonreir
- Un rostro sin movimiento por varios segundos podria ser una imagen.

### Variacion entre fotogramas

- Se capturan posiciones clave de la malla facial (cejas, ojos, labios).
- Si no hay cambios significativos en los puntos clave durante varios fotogramas consecutivos, se considerara que no hay interaccion humana real.

## Umbrales

Parametro                                    

- Movimiento entre puntos:  diferencia minima de 2–5 px por frame   
- Intervalo sin parpadeo:   2–8 segundos maximo sin parpadear       
- Expresiones detectadas:    cambios en angulos o distancias faciales
- Tiempo sin movimiento:   mas de 1.5–2 segundos sin variacion     

