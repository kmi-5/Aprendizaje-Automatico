# Clasificador de Noticias Reales vs Falsas con Naive Bayes

# Descripción
Modelito basico de clasificación automática que utiliza el algoritmo Naive Bayes para detectar si una noticia es real o falsa basándose en su contenido textual.

# Características
    - Procesamiento de texto con CountVectorizer
    - Implementación de Naive Bayes Multinomial
    - Evaluación de métricas de rendimiento
    - Clasificación de nuevas noticias

# Dataset
    El modelo se entrena con un conjunto de 10 noticias etiquetadas:
    - 5 noticias reales
    - 5 noticias falsas

# Estructura del Código

    1. Importación de Librerías
        ```python
        import pandas as pd
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        ```

    2. Preparación de Datos
    Carga del dataset de noticias etiquetadas
    Vectorización del texto usando CountVectorizer

    3. Entrenamiento del Modelo
    División de datos (70% entrenamiento, 30% prueba)
    Entrenamiento con Naive Bayes Multinomial

    4. Evaluación
    Cálculo de accuracy
    Matriz de confusión
    Reporte de clasificación

    5. Clasificación
    Predicción para nuevas noticias

        El modelo genera:
            Precisión total del modelo
            Matriz de confusión
            Reporte detallado (precision, recall, f1-score)
            Clasificación de nuevas noticias


# Requisitos:
        pandas
        scikit-learn
        Python 3.6+