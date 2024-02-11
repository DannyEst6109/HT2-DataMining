# Importar bibliotecas
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import matplotlib.pyplot as plt

# Cargar datos
datos = pd.read_csv("movies.csv", encoding='ISO-8859-1').dropna()

#1
# Seleccionar variables relevantes
variables_no_aportan = ["Id", "title", "homePage", "video", "director", "productionCompany", "productionCountry", "actors", "actorsCharacter"]
variables_aportan = ["popularity", "originalLanguage", "budget", "revenue", "runtime", "genres", "genresAmount", "productionCoAmount", "productionCountriesAmount", "releaseDate", "voteCount", "voteAvg", "actorsPopularity", "actorsAmount", "castWomenAmount", "castMenAmount"]

datos_procesados = datos[variables_aportan]

# Limpiar la columna 'releaseDate'
datos_procesados['releaseDate'] = pd.to_datetime(datos_procesados['releaseDate'], errors='coerce')
datos_procesados['releaseYear'] = datos_procesados['releaseDate'].dt.year
datos_procesados['releaseMonth'] = datos_procesados['releaseDate'].dt.month
datos_procesados['releaseDay'] = datos_procesados['releaseDate'].dt.day

# Eliminar variables no numéricas
datos_procesados = datos_procesados.select_dtypes(include=[np.number])

# Escalar los datos
scaler = StandardScaler()
clustering = scaler.fit_transform(datos_procesados)

#2
# Estadístico de Hopkins
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
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
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0

    return H

# Calcular estadístico de Hopkins
hopkins_statistic = hopkins(pd.DataFrame(clustering))
print("Estadístico de Hopkins:", hopkins_statistic)

# Análisis de tendencia al agrupamiento usando k-means y clustering jerárquico
wss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=100, random_state=0)
    kmeans.fit(clustering)
    wss.append(kmeans.inertia_)

#3
# Graficar codo
plt.plot(range(1, 11), wss, marker='o')
plt.xlabel('Cantidad de Clusters')
plt.ylabel('Suma de cuadrados dentro del grupo')
plt.show()

# Elegir el número óptimo de clusters
n_clusters = 6

#4
# Aplicar k-means
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
datos['grupo_kmeans'] = kmeans.fit_predict(clustering)

# Graficar resultados de k-means
plt.scatter(clustering[:, 0], clustering[:, 1], c=datos['grupo_kmeans'], cmap='rainbow')
plt.title('Resultados de K-means')
plt.show()

# Aplicar clustering jerárquico
linkage_matrix = linkage(clustering, method='ward')
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix)
plt.show()

# Cortar el dendrograma para obtener clusters
clusters_hc = cut_tree(linkage_matrix, n_clusters=n_clusters).flatten()
datos['grupo_hc'] = clusters_hc

# Graficar resultados de clustering jerárquico
plt.scatter(clustering[:, 0], clustering[:, 1], c=datos['grupo_hc'], cmap='rainbow')
plt.title('Resultados de Clustering Jerárquico')
plt.show()

#5
# Evaluar la calidad del agrupamiento con el método de la silueta
silueta_kmeans = silhouette_score(clustering, datos['grupo_kmeans'])
silueta_hc = silhouette_score(clustering, datos['grupo_hc'])

print(f"Silueta K-Means: {silueta_kmeans}")
print(f"Silueta Hierarchical Clustering: {silueta_hc}")


'''
# 6. Interpretar los grupos
# Descripción de grupos basada en medidas de tendencia central y tablas de frecuencia
grupo_medias = datos_procesados.groupby('grupo').mean()
grupo_moda = datos_procesados.groupby('grupo').apply(lambda x: x.mode().iloc[0])
grupo_mediana = datos_procesados.groupby('grupo').median()

# Visualización de resultados
print("Medias por Grupo:")
print(grupo_medias)
print("\nModa por Grupo:")
print(grupo_moda)
print("\nMedianas por Grupo:")
print(grupo_mediana)
'''