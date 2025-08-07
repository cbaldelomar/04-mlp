import numpy as np


class MLP:
    def __init__(self, layer_sizes):
        """
        layer_sizes: lista de enteros, donde el primer elemento es el tamaño de la capa de entrada,
        los elementos intermedios son los tamaños de las capas ocultas, y el último elemento es el tamaño
        de la capa de salida.
        """
        self.layer_sizes = layer_sizes

        self.weights = []  # Lista 3D: pesos entre capas (matrices numpy)
        self.biases = []  # Lista 3D: biases por capa (matrices numpy de 1 fila)

        self.activations_a = []  # Lista 2D: activaciones por capa
        self.interim_zs = []  # Lista 3D: valores lineales (z) para backpropagation

        self.__iniciar_pesos_y_biases(layer_sizes)

    def __iniciar_pesos_y_biases(self, layer_sizes):
        """
        Inicializa los pesos y sesgos de la red neuronal.
        Los pesos se inicializan con valores aleatorios muy pequeños y cercanos a cero.
        Los sesgos se inicializan con valores aleatorios muy pequeños y cercanos a cero.
        """
        for i in range(len(layer_sizes) - 1):
            # Los pesos se organizan en una matriz de dimensiones (capa_actual x capa_siguiente)
            layer_weights = np.random.rand(layer_sizes[i], layer_sizes[i + 1]) * 0.01

            # Los sesgos son un único vector por capa, codificados en una fila
            layer_biases = np.random.rand(1, layer_sizes[i + 1]) * 0.01

            # Se agregan a las listas correspondientes
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)

    def feedforward(self, inputs):
        """
        Calcula la salida de la red neuronal para las entradas dadas.
        """
        # Inicializa las activaciones y añade las entradas como primera capa
        activation = inputs
        self.activations_a = []
        self.activations_a.append(activation)

        # Recorre las capas para calcular z y aplicar la activación
        for layer_weight, layer_bias in zip(self.weights, self.biases):
            z = np.dot(activation, layer_weight) + layer_bias
            activation = self.__sigmoid(z)
            self.activations_a.append(activation)

        return activation  # Devuelve la salida final como predicción

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, inputs, ground_truth, learning_rate, iterations):
        for _ in range(iterations):
            self.__backpropagation(inputs, ground_truth, learning_rate)

    def __backpropagation(self, inputs, Y_ground_truth, learning_rate):
        # 1. Paso Feedforward: obtenemos las predicciones de la red
        Y_hat_output = self.feedforward(inputs)

        # 2. Número de ejemplos en el conjunto de entrenamiento
        n = Y_ground_truth.shape[0]

        # 3. Inicialización de la lista de deltas por capa
        delta_capa = [None] * len(self.weights)

        # 4. Error en la capa de salida: derivada del costo respecto a la activación
        delta_A = self.__derive_C_respect_derive_A(Y_ground_truth, Y_hat_output)

        # 5. Derivada de la activación respecto a la suma ponderada (Z)
        delta_Z = self.__derive_A_respect_derive_Z(Y_hat_output)

        # 6. Delta final de la capa de salida = combinación de los deltas anteriores
        delta_capa[-1] = delta_A * delta_Z

        # 7. Retropropagación del error hacia las capas ocultas
        # se resta -2 por la capa de entrada que no tiene peso
        for l in range(len(delta_capa) - 2, -1, -1):
            # Derivada acumulada del error respecto a la activación anterior
            delta_A = self.__derive_E_respect_derive_A(delta_capa, l)

            # Derivada de la activación sigmoide en la capa actual
            delta_Z = self.__derive_A_respect_derive_Z(self.activations_a[l + 1])

            # Delta en capa oculta
            delta_capa[l] = delta_A * delta_Z

        # 8. Actualización de pesos y sesgos en cada capa
        for l in range(len(self.weights)):
            # 8.1. Ajuste de pesos con la derivada acumulada
            delta_W = self.__derive_Z_respect_derive_W(delta_capa, l)
            self.weights[l] -= learning_rate * delta_W

            # 8.2. Ajuste de biases con la derivada acumulada
            delta_B = self.__derive_Z_respect_derive_B(delta_capa, l)
            self.biases[l] -= learning_rate * delta_B

    def __derive_C_respect_derive_A(self, Y_ground_truth, Y_hat_output):
        """
        Calcula la derivada de la función de costo respecto a la activación
        de la capa de salida. En este caso se asume costo tipo MSE (Error Cuadrático Medio).
        """
        # / n (de alguna forma lo "sopesa" el learning rate)
        return -2 * (Y_ground_truth - Y_hat_output)

    def __derive_A_respect_derive_Z(self, f_sigmoide):
        """
        Calcula la derivada de la activación sigmoide respecto a Z.
        Es decir: f'(z) = f(z) * (1 - f(z))
        """
        return f_sigmoide * (1 - f_sigmoide)

    def __derive_Z_respect_derive_W(self, delta_capa, l):
        """
        Calcula el gradiente de Z con respecto a los pesos en la capa l.
        Devuelve una matriz que representa cómo cambia Z cuando se ajustan los pesos.
        """
        return np.dot(self.activations_a[l].T, delta_capa[l])

    def __derive_Z_respect_derive_B(self, delta_capa, l):
        """
        Calcula el gradiente de Z con respecto a los biases en la capa `l`.

        Este gradiente representa cómo una pequeña variación en los biases afecta
        la suma ponderada (Z) de las neuronas. Se obtiene sumando los deltas
        de error de todas las muestras en la capa correspondiente. Este valor se
        utiliza durante el proceso de retropropagación para ajustar los biases.

        Parámetros:
            delta_capa (list of np.ndarray): Lista con los deltas de cada capa.
            l (int): Índice de la capa actual.

        Retorna:
            np.ndarray: Gradiente acumulado respecto a los biases de la capa `l`.
        """
        return np.sum(delta_capa[l], axis=0, keepdims=True)

    def __derive_E_respect_derive_A(self, delta_capa, l):
        """
        Calcula la derivada del error en la capa `l+1` respecto a la activación de la capa `l`.

        Este paso es crucial en la retropropagación. Se realiza multiplicando el delta de la
        capa siguiente por la transpuesta de los pesos que conectan la capa `l` con `l+1`.

        Parámetros:
            delta_capa (list of np.ndarray): Lista que contiene el delta (error) por capa.
            l (int): Índice de la capa actual (capa anterior a la que origina el error).

        Retorna:
            np.ndarray: Matriz de error proyectado sobre la activación de la capa `l`.
        """
        return np.dot(delta_capa[l + 1], self.weights[l + 1].T)
