# Importar librerías
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

# Configuración de visualización
pd.set_option('display.max_columns', None)

# Cargar datos
datos = pd.read_csv("movies.csv", encoding='latin-1')
print(datos.head())


# 1. Preprocesamiento del dataset
# Variables que no aportan información
variables_no_aportan = ['id', 'title', 'homePage', 'video', 'director', 'productionCompany', 'productionCountry', 'actors', 'actorsCharacter']
datos_procesados = datos.drop(variables_no_aportan, axis=1)

# Variables que sí aportan información
variables_aportan = ['popularity', 'originalTitle', 'originalLanguage', 'budget', 'revenue', 'runtime', 'genres', 'genresAmount', 'productionCoAmount', 'productionCountriesAmount', 'releaseDate', 'voteCount', 'voteAvg', 'actorsPopularity', 'actorsAmount', 'castWomenAmount', 'castMenAmount']

#datos_procesados = datos.drop(variables_aportan, axis=1)

datos_procesados = datos_procesados[variables_aportan]

# 2. Análisis de la tendencia al agrupamiento
# Estadístico de Hopkins
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
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

# Hopkins Statistic
numericas = datos_procesados.drop(['originalTitle', 'originalLanguage', 'genres', 'releaseDate'], axis=1)

numericas = numericas.apply(pd.to_numeric, errors='coerce')

hopkins_statistic = hopkins(numericas)
numericas = numericas.dropna()

print("Hopkins Statistic:", hopkins_statistic)

# Visual Assessment of cluster Tendency (VAT)
from scipy.spatial.distance import pdist, squareform

dist_data = pdist(numericas, metric='euclidean')
dist_matrix = squareform(dist_data)
plt.figure(figsize=(12, 8))
plt.imshow(dist_matrix, cmap='pink', interpolation='none')
plt.show()

# 3. Determinar el número de grupos más adecuado
# Gráfica de codo
wss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=100)
    kmeans.fit(numericas)
    wss.append(kmeans.inertia_)

plt.plot(range(1, 11), wss, marker='o')
plt.title('Gráfica de Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('Suma de Cuadrados Intra-Cluster')
plt.show()

# Elección del número de clusters
n_clusters = 6

# 4. Utilizar algoritmos K-medias y clustering jerárquico
# K-means
kmeans = KMeans(n_clusters=n_clusters, max_iter=100)
datos_procesados['grupo'] = kmeans.fit_predict(numericas)

# Clustering Jerárquico
linkage_matrix = linkage(numericas, method='ward')
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix)
plt.show()

# 5. Calidad del agrupamiento con el método de la silueta
# K-means
silhouette_avg_kmeans = silhouette_score(numericas, datos_procesados['grupo'])
print("Silhouette Score (K-means):", silhouette_avg_kmeans)

# Visualización de la silueta para K-means
silhouette_values_kmeans = silhouette_samples(numericas, datos_procesados['grupo'])
y_lower = 10
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    cluster_silhouette_values = silhouette_values_kmeans[datos_procesados['grupo'] == i]
    cluster_silhouette_values.sort()
    size_cluster_i = cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = plt.cm.nipy_spectral(float(i) / n_clusters)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    y_lower = y_upper + 10

plt.title("Silueta para K-means")
plt.xlabel("Coeficiente de Silueta")
plt.ylabel("Etiqueta del Cluster")
plt.show()

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