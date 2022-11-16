# Adrián Hernández Young - B93825
# LAB 1

# Parte 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 
import seaborn as sns

def load_data():
    df = pd.read_csv('titanic.csv') # Se vuelve una matriz con las entradas y columnas del csv

    df = df.dropna() # Se eliminan las entradas con valores faltantes

    # Decidí eliminar la columna de Name y PassengerId, debido a que
    # considero que son repetitivos, además de que si ocupara el id de
    # algún pasajero, podría sacar en cuál fila se encuentra y listo.
    df.drop('Name', inplace=True, axis=1) # Se eliminan las columnas que no se creen importantes
    df.drop('PassengerId', inplace=True, axis=1)
    # La columna de Ticket no considero que sea necesaria debido a que son es para
    # identificar a personas especificas y yo quiero algo más general.
    df.drop('Ticket', inplace=True, axis=1)
    # La columna de Cabin tampoco considero que sea necesaria ya que identifica de manera muy específica
    # a los pasajeros y yo quiero algo más general.
    df.drop('Cabin', inplace=True, axis=1)
    
    df = pd.get_dummies(df,columns=['Embarked','Sex'])
    
    return df

df = load_data()


# Parte 2
matrix = df.to_numpy()
matrix = matrix.astype(float)


# Parte 3
class myPCA:
    def __init__(self, matrix):
        self.matrix = matrix
    
    def principal_components(self):
        
        # ----------------------- 3a -----------------------
        mean = np.mean(matrix, axis=0)
        strd = np.std(matrix, axis=0)
        
        matrix_x = self.matrix
        
        for r in range (len(matrix_x)):
            for c in range(len(matrix_x[0])):
                matrix_x[r][c] = (matrix_x[r][c] - mean[c]) / strd[c]
        
        # ----------------------- 3b -----------------------
        correlation_r = 1 / len(matrix_x) * np.matmul(matrix_x.T, matrix_x)
        
        # ----------------------- 3c -----------------------
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_r)
        
        # ----------------------- 3d -----------------------
        idx = eigenvalues.argsort()[::-1]
        sorted_eigenvalues = eigenvalues[idx] 
        sorted_eigenvectors_v = eigenvectors[:,idx]
        
        # ----------------------- 3e -----------------------
        matrix_c = np.matmul(matrix_x, sorted_eigenvectors_v)
        
        # ----------------------- 3f -----------------------
        columns = len(matrix_c[0])
        inertia = []
        for i in sorted_eigenvalues:
            inertia.append(i / columns)
        
        # ----------------------- 3g -----------------------
        # Noto que si no hago esta parte, las dimensiones de mi gráfico dan las mismas que las de sklearn, pero siempre
        # reflejado.
        for r in range(len(sorted_eigenvectors_v)):
            for c in range(2):
                sorted_eigenvectors_v[r][c] = sorted_eigenvectors_v[r][c] * math.sqrt(sorted_eigenvalues[c])
        
        plt.figure(figsize=(15,15))
        plt.axhline(0, color='b')
        plt.axvline(0, color='b')
        for i in range(0, df.shape[1]):
        	plt.arrow(0,0, sorted_eigenvectors_v[i, 0],  # x - PC1
                      	sorted_eigenvectors_v[i, 1],  # y - PC2
                      	head_width=0.05, head_length=0.05)
        	plt.text(sorted_eigenvectors_v[i, 0] + 0.05, sorted_eigenvectors_v[i, 1] + 0.05, df.columns.values[i])
        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an),color="b")  # Circle
        plt.axis('equal')
        plt.title('Correlation Circle')
        plt.show()
        
        # Parte 4

        plt.scatter(np.ravel(matrix_c[:,0]),np.ravel(matrix_c[:,1]),c = ['b' if i==1 else 'r' for i in df['Survived']])
        plt.xlabel('PCA 1 (%.2f%% inertia)' % (inertia[0],))
        plt.ylabel('PCA 2 (%.2f%% inertia)' % (inertia[0],))
        plt.title('PCA')
        plt.show()
        
        return sorted_eigenvectors_v
        
        
x = myPCA(matrix)

x2 = x.principal_components()

# Parte 5

#En el gráfico de la inercia, podemos notar que los puntos azules, en su mayoria se hicieron hacia
#el lado izquierdo, mientras que los rojos hacia el lado derecho.  Por lo tanto podemos notar 2 grupos.

#En el círculo de correlación se formaron varios grupitos, lo que indican relación entre las variables, tenemos:
#    1. 'Sex_female' y 'Survived'
#    2. 'Parch' y 'SibSp'
#    3. 'Embarked_S' y 'PClass'
#Por otro lado tenemos que los demás valores se encuentran bastante
#alejados y distribuidos unos de los otros, entonces los demás se encuentran solos apuntando hacia direcciones distintas.

#Se puede observar una correlación negativa entre 'Sex_male' y el grupo
#de 'Sex_female' y 'Survived', por lo que muestra que son 2 sexos distintos,
#para el cual la variable 'Survived' estaba más relacionada con si la persona
#era una mujer, por lo tanto, si era mujer, tenía más chance que los hombres
#de sobrevivir.


# Parte 6

#Si usted estuviera en el Titanic los atributos o características
#que maximizarían sus probabilidades de sobrevivir serían si:
#    1. Principalmente usted es una mujer, haber sido una mujer en 
#       esta situación parece haber tenido una enorme relación con 
#       si usted sobrevivía o no.
#    2. Seguido vienen a estar las variables Parch y SibSp, lo que
#       significa que de tener menos o ningún familiar o familiares
#       también aumentaba su chance de sobrevivir.
    
# Parte 7

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = load_data()
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

pca = PCA()
C = pca.fit_transform(df_scaled)

inertia = pca.explained_variance_ratio_
V = pca.transform(np.identity(df_scaled.shape[1]))

plt.figure(figsize=(15,15))
plt.axhline(0, color='b')
plt.axvline(0, color='b')
for i in range(0, df.shape[1]):
	plt.arrow(0,0, V[i, 0],  # x - PC1
              	V[i, 1],  # y - PC2
              	head_width=0.05, head_length=0.05)
	plt.text(V[i, 0] + 0.05, V[i, 1] + 0.05, df.columns.values[i])
an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an),color="b")  # Circle
plt.axis('equal')
plt.title('Correlation Circle')
plt.show()

   
plt.scatter(np.ravel(C[:,0]),np.ravel(C[:,1]),c = ['b' if i==1 else 'r' for i in df['Survived']])
plt.xlabel('PCA 1 (%.2f%% inertia)' % (inertia[0],))
plt.ylabel('PCA 2 (%.2f%% inertia)' % (inertia[0],))
plt.title('PCA')
plt.show()

# Parte 8

#Mis gráficos en comparación con las de sklearn sí presentan variaciones.

#Según lo que investigué, me parece que sklearn realiza unos cálculos de 
#manera distinta, lo que pudo haber causado las diferencias.  

#De igual manera, las diferencias presentadas parecen ser mínimas y no 
#de gran impacto a la hora de interpretar los resultados.  Esto es debido 
#a que mis gráficos parecen ser un reflejo de los de sklearn con respecto a 
#un eje, sin embargo muestran la misma figura.
#Además, la forma de leer estos gráficos no varía con la perspectiva que uno 
#los vea, tipo no importa si los valores se encuentran del lado izquierdo,
#derecho, arriba o abajo del gráfico, sino que importa ver los grupos que se crean de los 
#valores, porque eso nos muestra su relación y los grupos que se formaron en los gráficos de sklearn y en los  
#míos son los mismos, por lo que podemos ver que mis gráficos están 
#agrupando los valores de manera correcta y creando una distancia entre los
#demás valores.  Por lo tanto no impacta el resultado.
