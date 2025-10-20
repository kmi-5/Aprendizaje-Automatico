
# 1. Importar librerías
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Crear dataset
datos = {
    "mensaje": [
        "¿Cuántos años tenés? No le digas a nadie que hablamos.",
        "Hola, ¿cómo estás? ¿Querés jugar algo online?",
        "Sos muy linda, no le muestres esto a tus papás.",
        "¿Terminaste la tarea de matemáticas?",
        "Podés mandarme una foto tuya, pero que sea solo para mí.",
        "¿Querés que te ayude con el juego que no podías pasar?",
        "No le comentes a nadie que hablamos por acá, ¿sí?",
        "¿Te gustó la película de ayer?",
        "¿Estás solo en casa ahora?",
        "¿Querés hacer una videollamada para estudiar juntos?",
        "No le digas a tus padres que te escribí, es nuestro secreto.",
        "¿Cómo te fue en el examen de historia?",
        "Mandame una foto tuya, pero que nadie más la vea.",
        "¿Jugamos Minecraft esta tarde?",
        "Sos muy especial para mí, no le cuentes a nadie lo que hablamos.",
        "¿Querés que te pase los apuntes de biología?",
        "¿Podés mostrarme cómo estás vestida ahora?",
        "¿Tenés ganas de salir a andar en bici mañana?",
        "No hace falta que le digas a nadie que hablamos tanto.",
        "¿Querés que estudiemos juntos para el parcial?"
    ],
    "etiqueta": [
        "grooming", "no grooming", "grooming", "no grooming",
        "grooming", "no grooming", "grooming", "no grooming",
        "grooming", "no grooming", "grooming", "no grooming",
        "grooming", "no grooming", "grooming", "no grooming",
        "grooming", "no grooming", "grooming", "no grooming"
    ]
}

df = pd.DataFrame(datos)

# 3. Vectorización de texto
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["mensaje"])
y = df["etiqueta"]


# 4. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 5. Entrenar modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X_train, y_train)


# 6. Evaluar el modelo
y_pred = modelo.predict(X_test)

print("Precisión del modelo:", accuracy_score(y_test, y_pred))
print("\n Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred, labels=["grooming","no grooming"]))


# 7. Clasificación de nuevos mensajes
nuevos_mensajes = [
    "¿Podés mandarme una foto tuya? No se la muestres a nadie.",
    "¿Jugamos Minecraft esta tarde?",
    "No le digas a tus papás que hablamos, ¿sí?",
    "¿Terminaste el trabajo práctico de historia?"
]

X_nuevos = vectorizer.transform(nuevos_mensajes)
predicciones = modelo.predict(X_nuevos)

print("\n Clasificación de nuevos mensajes:")
for mensaje, etiqueta in zip(nuevos_mensajes, predicciones):
    print(f"'{mensaje}' -> {etiqueta}")
