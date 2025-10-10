# Segmentación de Jurisdicciones Argentinas usando K-Means

## Descripción General
Este proyecto implementa un análisis de segmentación utilizando el algoritmo K-Means para agrupar jurisdicciones argentinas según la cantidad de viviendas habitadas. El objetivo es identificar patrones de densidad poblacional y facilitar la planificación de políticas públicas.

## Fases del Proceso

1. **Preparación de Datos**
   - Carga del dataset jurisdiccional
   - Extracción de variable numérica

2. **Determinación de K Óptimo**
   - Cálculo de inercia para k=1-10
   - Análisis de reducción porcentual
   - Selección basada en método del codo

3. **Aplicación de K-Means**
   - Segmentación con k=3
   - Asignación de clusters
   - Cálculo de centroides

4. **Análisis de Resultados**
   - Estadísticas descriptivas por cluster
   - Identificación de valores extremos
   - Interpretación práctica

## Características del Proyecto

# **Dataset**
- **Tamaño**: 24 jurisdicciones argentinas
- **Variable principal**: 
  - `Viviendas Habitadas`: Cantidad total de viviendas ocupadas por jurisdicción
- **Alcance geográfico**: Todas las provincias y la Ciudad Autónoma de Buenos Aires

# **Tecnologías Utilizadas**
- **Python 3.x**
- **Librerías principales**:
  - `scikit-learn`: Para implementación de K-Means
  - `pandas`: Manipulación y análisis de datos
  - `matplotlib`: Visualización del método del codo

## Metodología Implementada

# 1 **Método del Codo para Determinación de K Óptimo**

inertia_values = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
- **Propósito**: Encontrar el número óptimo de clusters
- **Rango analizado**: k = 1 a k = 10
- **Métrica**: Inercia (suma de distancias al cuadrado)

# 2 **Análisis de Reducción de Inercia**

reduccion = ((inertia_values[i-1] - inertia_values[i]) / inertia_values[i-1]) * 100
- **Cálculo**: Reducción porcentual entre valores consecutivos de k
- **Interpretación**: Identificar el punto donde ganancias marginales disminuyen

# 3 **Segmentación Final**

kmeans_final = KMeans(n_clusters=k_optimo, random_state=42)
df['Cluster'] = kmeans_final.fit_predict(X)
- **K óptimo seleccionado**: 3 clusters
- **Reproducibilidad**: `random_state=42`

## Outputs Generados

# 1. **Gráfico del Método del Codo**
- Visualización de inercia vs número de clusters
- Identificación del punto de inflexión (codo)

# 2. **Segmentación en 3 Clusters**
- **Cluster 0**: Jurisdicciones con menor cantidad de viviendas
- **Cluster 1**: Jurisdicciones con cantidad intermedia de viviendas  
- **Cluster 2**: Jurisdicciones con mayor cantidad de viviendas

# 3. **Estadísticas Descriptivas por Cluster**
- Tamaño de cada grupo
- Rango de viviendas
- Promedio por cluster
- Listado de jurisdicciones incluidas

## Conclusiones identificadas

# Distribución Identificada:
- **Cluster de alta densidad**: Buenos Aires, CABA, Córdoba, Santa Fe
- **Cluster de densidad media**: Mendoza, Tucumán, entre otras
- **Cluster de baja densidad**: Provincias con menor población

# Aplicaciones Prácticas:
- Planificación de infraestructura diferenciada
- Asignación eficiente de recursos públicos
- Desarrollo de políticas habitacionales segmentadas
- Análisis de patrones de distribución poblacional

## Configuración y Ejecución

# Requisitos Previos
    pip install scikit-learn pandas matplotlib

# Ejecución
    python segmentacion-kmeans.py