def predecir_resultado(accion, duracion):
  
    if accion == "Combate":
        if duracion > 100:
            return "Victoria"
        else:
            return "Derrota"
    
    elif accion == "Exploración":
        if duracion > 250:
            return "Descubrimiento"
        else:
            return "Sin hallazgos"
    
    elif accion == "Interacción social":
        return "Mensaje enviado"
    
    else:
        return "Resultado desconocido"

accion = input("Ingrese la acción (Combate / Exploración / Interacción social): ")
duracion = int(input("Ingrese la duración en segundos: "))

resultado = predecir_resultado(accion, duracion)
print(f"El resultado predicho es: {resultado}")