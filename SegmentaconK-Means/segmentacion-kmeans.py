import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cargar los datos
data = {
    'Jurisdicción': [
        'Ciudad Autónoma de Buenos Aires', 'Buenos Aires', 'Catamarca', 'Chaco', 'Chubut', 'Córdoba',
        'Corrientes', 'Entre Ríos', 'Formosa', 'Jujuy', 'La Pampa', 'La Rioja', 'Mendoza', 'Misiones',
        'Neuquén', 'Río Negro', 'Salta', 'San Juan', 'San Luis', 'Santa Cruz', 'Santa Fe',
        'Santiago del Estero', 'Tierra del Fuego, Antártida e Islas del Atlántico Sur', 'Tucumán'
    ],
    'Viviendas Habitadas': [
        1391258, 5970702, 131978, 368728, 213317, 1378216,
        370958, 494473, 194689, 238141, 140879, 128149, 639467, 420101,
        254545, 276371, 404904, 241436, 182864, 118047, 1273460,
        311361, 65535, 493794
    ]
}

df = pd.DataFrame(data)

# Calcular el metodo del codo
X = df[['Viviendas Habitadas']]

# Aplicar el metodo del codo para encontrar el número óptimo de clústeres
inertia_values = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
    
# Trazar el grafico del codo
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o')
plt.title('Método del Codo para determinar K-Mean óptimo')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.grid(True)
plt.xticks(k_values)
plt.show()

# Conclusiones de los resultados obtenidos
print("\n" + "-"*60)
print("Conclusiones - Segmentación de Jurisdicciones por Viviendas Habitadas")
print("-"*60)

# Análisis del método del codo
print("\nAnalisis del Método del Codo")
print(f"Valores de inercia para k=1 a k=10:")
for k, inertia in zip(k_values, inertia_values):
    print(f"  k={k}: Inercia = {inertia:,.2f}")

# Calcular reducción porcentual de inercia
reducciones = []
for i in range(1, len(inertia_values)):
    reduccion = ((inertia_values[i-1] - inertia_values[i]) / inertia_values[i-1]) * 100
    reducciones.append(reduccion)

print(f"\nReducción porcentual de inercia entre k valores consecutivos:")
for i, reduccion in enumerate(reducciones, 1):
    print(f"  k={i}→k={i+1}: {reduccion:.2f}%")

# Determinar k optimo basado en el "codo"
k_optimo = 3

print(f"\nRecomendación de k óptimo:")
print(f"Basado en el análisis del gráfico del codo, se recomienda usar k = {k_optimo}")
print("A partir de este punto, la reducción de inercia se hace menos significativa")

# Aplicar K-Means con k optimo
kmeans_final = KMeans(n_clusters=k_optimo, random_state=42)
df['Cluster'] = kmeans_final.fit_predict(X)

# Obtener centroides
centroides = kmeans_final.cluster_centers_.flatten()

print(f"\nResultados de segmentacion (k={k_optimo})")
print(f"Centroides de cada cluster (viviendas habitadas):")
for i, centroide in enumerate(centroides):
    print(f"  Cluster {i}: {centroide:,.0f} viviendas")

# Analisis detallado por cluster
print(f"\nDistribucion de jurisdicciones por cluster")

for cluster_num in range(k_optimo):
    cluster_data = df[df['Cluster'] == cluster_num]
    jurisdicciones = cluster_data['Jurisdicción'].tolist()
    
    print(f"\nCluster {cluster_num} ({len(cluster_data)} jurisdicciones):")
    print(f"  Rango de viviendas: {cluster_data['Viviendas Habitadas'].min():,} - {cluster_data['Viviendas Habitadas'].max():,}")
    print(f"  Promedio de viviendas: {cluster_data['Viviendas Habitadas'].mean():,.0f}")
    print(f"  Jurisdicciones incluidas:")
    for juris in jurisdicciones:
        print(f"    - {juris}")

# Estadisticas generales
print(f"\nEstadisticas generales de viviendas habitadas")
print(f"Total de jurisdicciones analizadas: {len(df)}")
print(f"Total de viviendas habitadas en todas las jurisdicciones: {df['Viviendas Habitadas'].sum():,}")
print(f"Promedio de viviendas por jurisdicción: {df['Viviendas Habitadas'].mean():,.0f}")
print(f"Desviación estándar: {df['Viviendas Habitadas'].std():,.0f}")

# Identificar jurisdicciones extremas
max_viviendas = df.loc[df['Viviendas Habitadas'].idxmax()]
min_viviendas = df.loc[df['Viviendas Habitadas'].idxmin()]

print(f"\nJurisdicciones con valores extremos:")
print(f"Mayor cantidad de viviendas: {max_viviendas['Jurisdicción']} ({max_viviendas['Viviendas Habitadas']:,})")
print(f"Menor cantidad de viviendas: {min_viviendas['Jurisdicción']} ({min_viviendas['Viviendas Habitadas']:,})")

print(f"\nAplicaciones prácticas de la segmentación:")
print("La segmentación permite:")
print("• Identificar patrones de densidad poblacional por jurisdicción")
print("• Asignar recursos de manera más eficiente según el tamaño del cluster")
print("• Desarrollar políticas habitacionales diferenciadas por segmento")
print("• Planificar infraestructura y servicios según la escala de cada grupo")