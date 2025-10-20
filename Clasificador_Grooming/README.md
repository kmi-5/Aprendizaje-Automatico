# Clasificador Bayesiano de Datos Ficticios de Grooming

# Descripción

Este proyecto implementa un **clasificador Naive Bayes** para detectar posibles mensajes de **grooming** utilizando un conjunto de datos ficticios.
El objetivo es simular un modelo básico de detección de lenguaje inapropiado en conversaciones digitales, a partir de un conjunto de frases etiquetadas como “grooming” o “no grooming”.

# Funcionalidad

1. **Generación del dataset:**
   Se crea manualmente un conjunto de mensajes simulados con etiquetas binarias (“grooming” / “no grooming”).
2. **Vectorización del texto:**
   Se utiliza `CountVectorizer` para transformar los mensajes en representaciones numéricas (bolsa de palabras).
3. **Entrenamiento y prueba del modelo:**
   Los datos se dividen en conjuntos de entrenamiento y prueba (70/30) usando `train_test_split`.
4. **Modelo Naive Bayes:**
   Se entrena un clasificador **MultinomialNB** para distinguir entre mensajes de grooming y no grooming.
5. **Evaluación del rendimiento:**
   Se imprimen métricas como **accuracy**, **matriz de confusión** y **reporte de clasificación**.
6. **Clasificación de nuevos mensajes:**
   El modelo predice la categoría (“grooming” o “no grooming”) para nuevos ejemplos ingresados.

# Librerías utilizadas

* **pandas:** manejo y estructura de los datos en formato tabular.
* **scikit-learn:**

  * `CountVectorizer` para convertir texto a vectores numéricos.
  * `train_test_split` para dividir los datos.
  * `MultinomialNB` como modelo bayesiano.
  * `accuracy_score`, `confusion_matrix`, `classification_report` para métricas.

# Uso

Ejecutar el script para ver el proceso completo de entrenamiento, evaluación y clasificación.
El modelo mostrará la precisión obtenida y luego clasificará una lista de nuevos mensajes ficticios.
