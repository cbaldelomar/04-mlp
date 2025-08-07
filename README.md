# Red Neuronal Multicapa (MLP)

Este proyecto implementa una red neuronal multicapa (MLP) desde cero en Python para predecir el precio de viviendas usando el dataset [Housing.csv](data/Housing.csv). El código principal se encuentra en el notebook [notebooks/01-main.ipynb](notebooks/01-main.ipynb) y la lógica de la red neuronal está en [libs/neuronas.py](libs/neuronas.py).

## Estructura del Proyecto

```
.gitignore
data/v
    Housing.csv
libs/
    neuronas.py
notebooks/
    01-main.ipynb
README.md
```

## Descripción

- **libs/neuronas.py**: Implementa la clase [`MLP`](libs/neuronas.py) para redes densas, incluyendo inicialización, feedforward y retropropagación.
- **notebooks/01-main.ipynb**: Notebook principal para cargar datos, preprocesar, entrenar y evaluar la red.
- **data/Housing.csv**: Dataset de precios de viviendas.

## Uso

1. Instala las dependencias necesarias:
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - scikit-learn

2. Ejecuta el notebook [notebooks/01-main.ipynb](notebooks/01-main.ipynb) para entrenar y evaluar la red MLP.

## Ejemplo de Entrenamiento

En el notebook se exploran diferentes arquitecturas y parámetros de entrenamiento, por ejemplo:

```python
from neuronas import MLP

RNA_10HL = MLP(layer_sizes=[4, 10, 4, 1])
RNA_10HL.train(x_train, y_train, learning_rate=0.05, iterations=50)
predicciones = RNA_10HL.feedforward(x_test)
```

## Licencia

Este proyecto es solo para fines educativos.