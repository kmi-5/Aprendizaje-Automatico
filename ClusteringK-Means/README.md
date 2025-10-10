# Clustering de Estudiantes Universitarios

## Descripción General
Este proyecto implementa un algoritmo de clustering utilizando K-Means para analizar y segmentar estudiantes universitarios basándose en tres características principales: edad, horas de estudio y promedio académico. El objetivo es identificar patrones y grupos naturales dentro de la población estudiantil.

## Características del Proyecto

# **Dataset**
- **Tamaño**: 50 estudiantes
- **Variables**:
  - `edad`: Edad del estudiante (años)
  - `horas_estudio`: Horas de estudio semanales
  - `promedio_academico`: Calificación académica promedio (escala 0-10)

# **Tecnologías Utilizadas**
- **Python 3.x**
- **Librerías principales**:
  - `scikit-learn`: Para implementación de K-Means y preprocesamiento
  - `pandas`: Manipulación y análisis de datos
  - `numpy`: Operaciones numéricas
  - `matplotlib`: Visualización de resultados

## Estructura del Código

# 1 **Preprocesamiento de Datos**
    `Estandarización de características`
    scaler = StandardScaler()
    X_estandarizado = scaler.fit_transform(X)

- **Propósito**: Normalizar las variables para que tengan media 0 y desviación estándar 1
- **Importancia**: K-Means es sensible a la escala de las variables

# 2. **Algoritmo de Clustering**
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_estandarizado)

-    **Parámetros**:
  - `n_clusters=4`: Número de grupos a identificar
  - `random_state=42`: Semilla para reproducibilidad
  - `n_init=10`: Número de inicializaciones para evitar mínimos locales

# 3 **Visualización**
- **Gráfico de dispersión**: Edad vs Promedio Académico
- **Centroides**: Marcados con 'X' rojas
- **Colores**: Diferentes clusters representados con colores distintos

# 4 **Análisis de Resultados**
- Estadísticas descriptivas por cluster
- Interpretación de perfiles estudiantiles
- Comparación con estadísticas generales

## Resultados Esperados

# **Outputs Generados**
1. **Visualización gráfica** de los clusters identificados
2. **Estadísticas por cluster**:
   - Tamaño del grupo
   - Edad promedio
   - Horas de estudio promedio
   - Rendimiento académico promedio
3. **Interpretación cualitativa** de los perfiles estudiantiles

# **Aplicaciones Prácticas**
- Identificación de patrones de comportamiento estudiantil
- Diseño de estrategias educativas personalizadas
- Detección de grupos con necesidades específicas
- Análisis de la relación entre edad, horas de estudio y rendimiento

## Configuración y Ejecución

# Requisitos Previos
    pip install scikit-learn pandas numpy matplotlib

# Ejecución
    python Clustering-Estudiantes.py
