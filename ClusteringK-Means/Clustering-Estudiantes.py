import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Carga de datos de estudiantes universitarios
estudiantes = [
    {'edad': 42, 'horas_estudio': 2, 'promedio_academico': 8.87},
    {'edad': 26, 'horas_estudio': 8, 'promedio_academico': 9.11},
    {'edad': 56, 'horas_estudio': 5, 'promedio_academico': 7.29},
    {'edad': 29, 'horas_estudio': 14, 'promedio_academico': 7.27},
    {'edad': 43, 'horas_estudio': 8, 'promedio_academico': 8.5},
    {'edad': 34, 'horas_estudio': 10, 'promedio_academico': 4.45},
    {'edad': 51, 'horas_estudio': 3, 'promedio_academico': 6.12},
    {'edad': 22, 'horas_estudio': 12, 'promedio_academico': 4.33},
    {'edad': 47, 'horas_estudio': 6, 'promedio_academico': 7.0},
    {'edad': 38, 'horas_estudio': 9, 'promedio_academico': 4.75},
    {'edad': 60, 'horas_estudio': 2, 'promedio_academico': 9.0},
    {'edad': 31, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 45, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 27, 'horas_estudio': 13, 'promedio_academico': 3.2},
    {'edad': 50, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 36, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 40, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 24, 'horas_estudio': 15, 'promedio_academico': 3.0},
    {'edad': 55, 'horas_estudio': 3, 'promedio_academico': 9.0},
    {'edad': 33, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 48, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 28, 'horas_estudio': 12, 'promedio_academico': 2.0},
    {'edad': 52, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 30, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 46, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 35, 'horas_estudio': 9, 'promedio_academico': 4.0},
    {'edad': 41, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 23, 'horas_estudio': 14, 'promedio_academico': 1.0},
    {'edad': 53, 'horas_estudio': 3, 'promedio_academico': 8.0},
    {'edad': 32, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 49, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 25, 'horas_estudio': 13, 'promedio_academico': 2.0},
    {'edad': 54, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 37, 'horas_estudio': 8, 'promedio_academico': 4.0},
    {'edad': 39, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 21, 'horas_estudio': 15, 'promedio_academico': 9.0},
    {'edad': 58, 'horas_estudio': 2, 'promedio_academico': 9.0},
    {'edad': 44, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 29, 'horas_estudio': 12, 'promedio_academico': 2.0},
    {'edad': 50, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 34, 'horas_estudio': 9, 'promedio_academico': 3.0},
    {'edad': 42, 'horas_estudio': 7, 'promedio_academico': 6.0},
    {'edad': 26, 'horas_estudio': 14, 'promedio_academico': 3.0},
    {'edad': 57, 'horas_estudio': 3, 'promedio_academico': 9.0},
    {'edad': 30, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 47, 'horas_estudio': 4, 'promedio_academico': 7.0},
    {'edad': 31, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 45, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 27, 'horas_estudio': 13, 'promedio_academico': 1.0},
    {'edad': 55, 'horas_estudio': 5, 'promedio_academico': 8.0}
]

# Crear el dataframe
df = pd.DataFrame(estudiantes)
X = df[['edad', 'horas_estudio', 'promedio_academico']].values

# Estandarizar los datos para aplicar k-means correctamente
scaler = StandardScaler()
X_estandarizado = scaler.fit_transform(X)

# Aplicar k-means con la cantidad de 4 clusters (K=4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_estandarizado)

# Obtener las etiquetas de los clusters y los centroides
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Transformar centroides de vuelta a la escala original para visualización
centroids_original = scaler.inverse_transform(centroids)

# Visualizar los resultados (edad vs promedio)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 2], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.scatter(centroids_original[:, 0], centroids_original[:, 2], c='red', s=200, marker='X', label='Centroides', edgecolors='black')

plt.title("Agrupamiento Estudiantes Universitarios K-Means (K=4)")
plt.xlabel("Edad simulada")
plt.ylabel("Promedio Académico")
plt.legend()
plt.grid(True)
plt.show()


# Conclusiones de los resultados obtenidos
print("\n" + "-"*60)
print("Conclusiones - Clustering de Estudiantes Universitarios")
print("-"*60)
print(f"Total de estudiantes analizados: {len(df)}")
print(f"Número de clusters: 4")

# Analisis por cada cluster
print("\nCaracteristicas por cluster")
df['cluster'] = labels
for cluster_num in range(4):
    cluster_data = df[df['cluster'] == cluster_num]
    print(f"\nCluster {cluster_num} ({len(cluster_data)} estudiantes):")
    print(f"  Edad promedio: {cluster_data['edad'].mean():.1f} años")
    print(f"  Horas estudio promedio: {cluster_data['horas_estudio'].mean():.1f} hrs/semana")
    print(f"  Promedio académico: {cluster_data['promedio_academico'].mean():.2f}")
    print(f"  Rango edad: {cluster_data['edad'].min()}-{cluster_data['edad'].max()} años")

# Interpretaciones de los clusters y estadisticas
print("\nEstadísticas generales")
print(f"Edad promedio total: {df['edad'].mean():.1f} años")
print(f"Horas estudio promedio total: {df['horas_estudio'].mean():.1f} hrs/semana")
print(f"Promedio académico total: {df['promedio_academico'].mean():.2f}")

print("\nInterpretación de clusters")
print("Cada cluster representa un perfil diferente de estudiante:")
print("• Basado en edad, horas de estudio y rendimiento académico")
print("• Los centroides representan el 'estudiante promedio' de cada grupo")
print("• Útil para identificar patrones y diseñar estrategias educativas personalizadas")