# Árbol de Decisión - Predicción de Aceptación de Ofertas de Telecomunicaciones

## Descripción del Proyecto

Este proyecto desarrolla un árbol de decisión para predecir si un cliente de telecomunicaciones aceptará una oferta de plan de datos móviles. El modelo utiliza técnicas de machine learning y teoría de la información para construir un clasificador basado en atributos demográficos y de uso del cliente.

## Objetivos

1. Calcular la entropía del conjunto de datos original
2. Evaluar la ganancia de información de diferentes atributos
3. Construir un árbol de decisión óptimo
4. Identificar el mejor atributo para comenzar la segmentación

## Datos

### Atributos Utilizados
- **Edad** (en años, agrupada en rangos)
- **Nivel de uso mensual de datos** (en GB, agrupado en rangos)
- **Tiene línea fija** (Sí/No)
- **Aceptó la oferta** (Variable objetivo - Sí/No)

### Rangos Definidos
- **Edad**: Joven (≤30), Adulto (31-50), Mayor (>50)
- **Uso de datos**: Bajo (≤3GB), Medio (3.1-6GB), Alto (>6GB)

## Tecnologías Utilizadas

- **Python 3.x**
- **pandas** - Manipulación de datos
- **scikit-learn** - Machine learning
- **matplotlib** - Visualización
- **math** - Cálculos matemáticos

# Estructura del Código

    ## Funciones Principales
    ```bash
    1. **`calcular_entropia(conjunto)`**
    - Calcula la entropía de un conjunto de datos
    - Utiliza la fórmula: `-Σ p_i * log2(p_i)`

    2. **`crear_rangos_edad(edad)`**
    - Clasifica la edad en rangos predefinidos

    3. **`crear_rangos_uso(uso)`**
    - Clasifica el uso de datos en rangos predefinidos

    4. **`calcular_ganancia_informacion(df, atributo, objetivo)`**
    - Calcula la ganancia de información para un atributo específico
    - Utiliza: `Ganancia = Entropía_total - Entropía_ponderada`
    ```

    ## Flujo del código

    1. **Carga y preparación de datos**
    2. **Cálculo de entropía inicial**
    3. **Evaluación de ganancias de información**
    4. **Construcción del árbol de decisión**
    5. **Visualización y evaluación**

    ## Resultados de Salida
        ```bash
        ## Entropía del Conjunto Original
        - **Valor**: 1.000 bits (máxima incertidumbre)

        ## Ganancias de Información
        | Tiene línea fija | **0.722** |
        | Edad (rangos) | 0.125 |
        | Uso de datos (rangos) | 0.125 |

        ## Mejor Atributo para el Árbol
        - **Atributo**: "Tiene línea fija"
        - **Ganancia**: 0.722
        ```

# Reglas de Decisión

    El árbol genera las siguientes reglas de predicción:

    1. **Si el cliente NO tiene línea fija** → NO acepta la oferta
    2. **Si el cliente SÍ tiene línea fija** → SÍ acepta la oferta
    3. **Atributos adicionales** (edad y uso de datos) refinan la predicción en casos específicos

    # Precisión del modelo:

    - **Precisión en entrenamiento**: 100%
    - **Tamaño del conjunto**: 10 clientes

#  Visualizaciones

El proyecto genera las siguientes visualizaciones:

1. **Árbol de decisión completo** con divisiones y valores de entropía
2. **Matriz de predicciones** comparando valores reales vs predichos

# ¿Cómo Ejecutar?

    ## Requisitos Previos
    ```bash
    pip install pandas scikit-learn matplotlib
    ```

    ## Ejecución
    ```bash
    python arbol_decision_telecom.py
    ``` 

# Interpretación de Resultados

    ## Para la Empresa
    1. **Segmentación efectiva**: El atributo "Tiene línea fija" es el mejor predictor
    2. **Estrategia de marketing**: Enfocar esfuerzos en clientes con línea fija existente
    3. **Optimización de recursos**: Reducir campañas en segmentos de baja conversión

    ## Técnica
    1. **Entropía alta inicial** indica diversidad en las respuestas
    2. **Ganancia significativa** en "Tiene línea fija" sugiere fuerte correlación
    3. **Árbol simple pero efectivo** para la segmentación inicial
