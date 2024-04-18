class ActivationFunctions:
    @staticmethod
    def sigmoid(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the sigmoid activation for a given input Z.

        Parameters:
        Z (np.ndarray): Input to the sigmoid function.

        Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the output of the sigmoid function (A) and the cache (Z).
        """
        A = 1 / (1 + np.exp(-Z))
        cache = Z
        return A, cache

    @staticmethod
    def sigmoid_derivative(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the sigmoid activation function.

        Parameters:
        dA (np.ndarray): Derivative of the cost function with respect to the activation.
        cache (np.ndarray): Cached input Z.

        Returns:
        np.ndarray: Derivative of the cost function with respect to the input Z.
        """
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        assert dZ.shape == Z.shape
        return dZ

    @staticmethod
    def relu(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the ReLU activation for a given input Z.

        Parameters:
        Z (np.ndarray): Input to the ReLU function.

        Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the output of the ReLU function (A) and the cache (Z).
        """
        A = np.maximum(0, Z)
        cache = Z
        assert A.shape == Z.shape
        return A, cache

    @staticmethod
    def relu_derivative(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the ReLU activation function.

        Parameters:
        dA (np.ndarray): Derivative of the cost function with respect to the activation.
        cache (np.ndarray): Cached input Z.

        Returns:
        np.ndarray: Derivative of the cost function with respect to the input Z.
        """
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert dZ.shape == Z.shape
        return dZ

    @staticmethod
    def tanh(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the hyperbolic tangent (tanh) activation for a given input Z.

        Parameters:
        Z (np.ndarray): Input to the tanh function.

        Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the output of the tanh function (A) and the cache (Z).
        """
        A = np.tanh(Z)
        cache = Z
        return A, cache

    @staticmethod
    def tanh_derivative(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the hyperbolic tangent (tanh) activation function.

        Parameters:
        dA (np.ndarray): Derivative of the cost function with respect to the activation.
        cache (np.ndarray): Cached input Z.

        Returns:
        np.ndarray: Derivative of the cost function with respect to the input Z.
        """
        Z = cache
        s = np.tanh(Z)
        dZ = dA * (1 - np.power(s, 2))
        assert dZ.shape == Z.shape
        return dZ
