# Importar bibliotecas
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.metrics import silhouette_samples, silhouette_score

# Configuración para mostrar gráficos en línea
# %matplotlib inline

# Cargar datos
datos = pd.read_csv("movies.csv", encoding='ISO-8859-1')
print(datos.head())

# Seleccionar variables relevantes
variables_no_aportan = ["id", "title", "homePage", "video", "director", 
                        "productionCompany", "productionCountry", "actors", "actorsCharacter"]

variables_aportan = ["popularity", "originalLanguage", "budget", "revenue", "runtime", 
                     "genres", "genresAmount", "productionCoAmount", "productionCountriesAmount", 
                     "releaseDate", "voteCount", "voteAvg", "actorsPopularity", 
                     "actorsAmount", "castWomenAmount", "castMenAmount"]

datos_procesados = datos[variables_aportan]

# Análisis de tendencia al agrupamiento utilizando el estadístico de Hopkins
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan

def hopkins(X):
    d = X.shape[1]
    n = len(X) 
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X, axis=0), d).reshape(1, -1), 2,
                                    return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0

    return H

# Preprocesamiento del dataset y análisis de tendencia al agrupamiento
numericas = datos_procesados.drop([ 'originalLanguage', 'genres', 'releaseDate'], axis=1)
numericas = numericas.apply(pd.to_numeric, errors='coerce')
numericas = numericas.dropna()
hopkins_statistic = hopkins(numericas)

# Determinar el número de grupos más adecuado con el método de codo
wss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=100, random_state=42)
    kmeans.fit(scale(numericas))
    wss.append(kmeans.inertia_)

# Graficar el método de codo
plt.plot(range(1, 11), wss, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Suma de Cuadrados Intra-Cluster')
plt.title('Método de Codo para Determinar Número de Clusters')
plt.show()

# Elegir el número de clusters
n_clusters = 6

# Aplicar K-means y asignar resultados directamente a una nueva columna en datos_procesados
grupo_kmeans = KMeans(n_clusters=n_clusters, max_iter=100, random_state=42).fit_predict(scale(numericas))

# Asegurarnos de que grupo_kmeans tiene la misma longitud que datos_procesados
if len(grupo_kmeans) == len(datos_procesados):
    datos_procesados['grupo_kmeans'] = grupo_kmeans
else:
    print("Error: Las longitudes no coinciden")

# Crear una nueva columna en datos_procesados y asignarle grupo_kmeans
datos_procesados['grupo_kmeans'] = grupo_kmeans

# Aplicar clustering jerárquico
linkage_matrix = linkage(scale(numericas), method='ward')
cut_tree_result = cut_tree(linkage_matrix, n_clusters=n_clusters).flatten()
datos_procesados['grupo_jerarquico'] = cut_tree_result

# Calidad del agrupamiento con el método de la silueta para K-means
silhouette_kmeans = silhouette_score(scale(numericas), grupo_kmeans)
print("Calidad del agrupamiento K-means (silueta):", silhouette_kmeans)

# Calidad del agrupamiento con el método de la silueta para clustering jerárquico
silhouette_jerarquico = silhouette_score(scale(numericas), datos_procesados['grupo_jerarquico'])
print("Calidad del agrupamiento Clustering Jerárquico (silueta):", silhouette_jerarquico)

# Interpretación de los grupos
# (continuación del código original, adaptado según sea necesario)

# Descripción del trabajo futuro
# (continuación del código original, adaptado según sea necesario)