
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar los datos desde el CSV especificando que la primera fila es un encabezado
df = pd.read_csv('datos_entrenamiento.csv')
# Crear la figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Graficar puntos con salto = 0
ax.scatter(df[df['salto'] == 0]['velocidad'], df[df['salto'] == 0]['distancia'], df[df['salto'] == 0]['salto'],
           c='blue', marker='o', label='No salto (0)', alpha=0.6)

# Graficar puntos con salto = 1
ax.scatter(df[df['salto'] == 1]['velocidad'], df[df['salto'] == 1]['distancia'], df[df['salto'] == 1]['salto'],
           c='red', marker='x', label='Salto (1)', alpha=0.6)
# Etiquetas de los ejes
ax.set_xlabel('Velocidad de la bola')
ax.set_ylabel('Distancia del jugador')
ax.set_zlabel('Salto?')
# Mostrar leyenda
ax.legend()
# Mostrar el gráfico
plt.show()




