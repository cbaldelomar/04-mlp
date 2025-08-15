# Red Neuronal Multicapa (MLP)

Este proyecto implementa una red neuronal multicapa (MLP) desde cero en Python para predecir el precio de viviendas usando el dataset [Housing.csv](data/Housing.csv). El código principal se encuentra en el notebook [notebooks/01-main.ipynb](notebooks/01-main.ipynb) y la lógica de la red neuronal está en [libs/neuronas.py](libs/neuronas.py).

El proyecto también incluye el notebook [notebooks/02-keras.ipynb](notebooks/02-keras.ipynb), que utiliza Keras/TensorFlow para construir y entrenar redes densas de manera profesional.

## Estructura del Proyecto

```
.gitignore
data/v
    Housing.csv
libs/
    neuronas.py
notebooks/
    01-main.ipynb
    02-keras.ipynb
README.md
```

## Descripción

- **libs/neuronas.py**: Implementa la clase [`MLP`](libs/neuronas.py) para redes densas, incluyendo inicialización, feedforward y retropropagación.
- **notebooks/01-main.ipynb**: Notebook principal para cargar datos, preprocesar, entrenar y evaluar la red.
- **data/Housing.csv**: Dataset de precios de viviendas.
- **notebooks/02-keras.ipynb**: Ejemplo completo de regresión con redes densas usando Keras/TensorFlow. Incluye:
    - Preprocesamiento avanzado con `Pipeline` y `ColumnTransformer`.
    - Modelos con diferentes arquitecturas (ej: 128-64, 64-32-16).
    - Entrenamiento con callbacks (`EarlyStopping`, `ModelCheckpoint`, `TensorBoard`).
    - Evaluación con métricas MAE, RMSE, R² y comparación con baseline.
    - Guardado y carga de modelos y pipelines para inferencia.

## Uso


1. Instala las dependencias necesarias.

    - Usar conda y el archivo `environment.yml` incluido en el repositorio:
        ```bash
        conda env create -f environment.yml
        conda activate tf_env
        ```

    - Alternativamente, puedes instalar manualmente:
        - numpy
        - pandas
        - matplotlib
        - seaborn
        - scikit-learn
        - tensorflow
        - ipykernel

2. Ejecuta el notebook [notebooks/01-main.ipynb](notebooks/01-main.ipynb) para entrenar y evaluar la red MLP.

3. Ejecuta el notebook [notebooks/02-keras.ipynb](notebooks/02-keras.ipynb) para entrenar modelos densos con Keras/TensorFlow y comparar resultados.


## Licencia

Este proyecto es solo para fines educativos.