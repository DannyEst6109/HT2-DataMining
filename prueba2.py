import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


# Cargar datos
datos = pd.read_csv("movies.csv", encoding='ISO-8859-1')

# Eliminar filas con valores perdidos
datos = datos.dropna()

# Variables que no aportan información
variables_no_aportan = ['id', 'title', 'homePage', 'video', 'director', 'productionCompany', 'productionCountry', 'actors', 'actorsCharacter']

# Variables que sí aportan información
variables_aportan = ['popularity', 'originalTitle', 'originalLanguage', 'budget', 'revenue', 'runtime', 'genres', 'genresAmount', 'productionCoAmount', 'productionCountriesAmount', 'releaseDate', 'voteCount', 'voteAvg', 'actorsPopularity', 'actorsAmount', 'castWomenAmount', 'castMenAmount']


# Preprocesamiento
datos_procesados = datos[variables_aportan]

# Convertir 'releaseDate' a tipo de dato datetime
datos_procesados['releaseDate'] = pd.to_datetime(datos_procesados['releaseDate'], errors='coerce')

# Extraer componentes temporales relevantes
datos_procesados['releaseYear'] = datos_procesados['releaseDate'].dt.year
datos_procesados['releaseMonth'] = datos_procesados['releaseDate'].dt.month
datos_procesados['releaseDay'] = datos_procesados['releaseDate'].dt.day

# Eliminar columnas originales
datos_procesados = datos_procesados.drop(['releaseDate'], axis=1)

# Definir columnas numéricas y categóricas para el preprocesamiento
columnas_numericas = datos_procesados.drop(['originalTitle', 'originalLanguage', 'genres'], axis=1).columns
columnas_categoricas = ['originalLanguage', 'genres']

# Crear transformador para aplicar CountVectorizer a columnas categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columnas_numericas),
        ('cat', CountVectorizer(tokenizer=lambda x: x.split('|')), 'genres')
    ],
    remainder='passthrough'  # Mantener columnas no especificadas sin transformación
)

# Crear pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

# Aplicar transformador
clustering_data = pipeline.fit_transform(datos_procesados)


# Determinar número de grupos con método de codo
wss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(clustering_data)
    wss.append(kmeans.inertia_)

# Gráfico de codo
plt.plot(range(1, 11), wss, marker='o')
plt.xlabel('Cantidad de Clusters')
plt.ylabel('Suma de cuadrados')
plt.title('Método de codo para determinar el número de clusters')
plt.show()

# Elegir el número de clusters
num_clusters = 6

# Algoritmo K-means
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
datos_procesados['grupo'] = kmeans_model.fit_predict(clustering_data)

# Algoritmo Jerárquico
linkage_matrix = linkage(clustering_data, method='ward')
dendrogram(linkage_matrix)
plt.title('Dendrograma para Clustering Jerárquico')
plt.show()

# Calidad del agrupamiento con método de silueta
silhouette_kmeans = silhouette_score(clustering_data, datos_procesados['grupo'])
print("Silueta para K-means:", silhouette_kmeans)

# Calcular la silueta para cada punto y graficar
silhouette_samples_kmeans = silhouette_samples(clustering_data, datos_procesados['grupo'])
plt.figure(figsize=(8, 6))
plt.scatter(range(len(datos_procesados)), silhouette_samples_kmeans, c=datos_procesados['grupo'], cmap='viridis')
plt.title('Silueta para K-means')
plt.show()

# Interprete los grupos (medias, modas, medianas)
for variable in variables_aportan:
    print(f"Media de {variable}: {np.mean(datos_procesados[variable]):.4f}")
