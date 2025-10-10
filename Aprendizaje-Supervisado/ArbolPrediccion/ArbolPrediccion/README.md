# Proyecto: Predicción de Abandono Estudiantil mediante Árboles de Decisión

Este proyecto implementa un modelo de **árbol de decisión** para predecir el riesgo de abandono académico de estudiantes, utilizando datos institucionales. Consta de dos scripts complementarios que realizan el entrenamiento del modelo y la generación de predicciones finales.


## Estructura del proyecto

Proyecto_Arbol_Prediccion/
│
├── ArbolEntrenamiento.py
├── ArbolFinal.py
├── TablaPrediccionAbandono-Entrenamiento.xlsx
├── TablaPrediccionAbandono-DatosFinal.xlsx
└── Predicciones_Final.xlsx  # (se genera al ejecutar ArbolFinal.py)


## Requisitos

Antes de ejecutar los scripts, asegurate de tener instaladas las siguientes librerías de Python:
    pip install pandas matplotlib seaborn scikit-learn openpyxl


## Descripción de los scripts

### 1 ArbolEntrenamiento.py

* Carga los datos del archivo `TablaPrediccionAbandono-Entrenamiento.xlsx`.
* Realiza limpieza de valores nulos y codificación de variables categóricas.
* Divide el dataset en conjuntos de entrenamiento y prueba.
* Entrena un modelo **DecisionTreeClassifier** con los siguientes parámetros:

  * `criterion='entropy'`
  * `max_depth=6`
  * `min_samples_leaf=5`
* Genera visualizaciones:

  * Matriz de confusión.
  * Distribución del estado final.
  * Importancia de variables.
  * Árbol de decisión.
* Imprime en consola una interpretación de resultados y recomendaciones.

### 2 ArbolFinal.py

* Reutiliza el codigo del modelo para predecir nuevos casos.
* Carga datos desde:

  * `TablaPrediccionAbandono-Entrenamiento.xlsx` (para entrenar).
  * `TablaPrediccionAbandono-DatosFinal.xlsx` (para predecir).
* Evalúa el rendimiento del modelo con matriz de confusión, precisión y reporte de clasificación.
* Genera un nuevo archivo `Predicciones_Final.xlsx` con una columna adicional: **PrediciónAbandono** (valores “Sí” o “No”).


## Resultados esperados

El modelo identifica las variables más influyentes en el abandono académico, tales como:

* Cantidad de materias aprobadas.
* Asistencia.
* Desempeño académico.

Estas variables permiten construir estrategias de prevención y apoyo a estudiantes en riesgo.


## Ejecución

Ejecutar los scripts desde la consola o un entorno como VSCode o Anaconda Prompt:

    python ArbolEntrenamiento.py
    python ArbolFinal.py

Los gráficos se mostrarán automáticamente y el archivo `Predicciones_Final.xlsx` se generará en la misma carpeta.