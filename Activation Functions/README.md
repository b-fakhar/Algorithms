### Activation Functions
ActivationFunctions is a Python class that provides implementations of commonly used activation functions for neural networks. This class offers methods to compute activation functions such as sigmoid, ReLU (Rectified Linear Unit), and hyperbolic tangent (tanh), as well as their derivatives for backpropagation.

##### Sigmoid:
  - $A = \frac{1}{1 + e^{-z}}$
  - $\frac{\partial A}{\partial Z} = A \cdot (1 - A)$

##### ReLU (Rectified Linear Unit) Function:
 - $A = \max(0, z)$
 - $\frac{\partial A}{\partial Z} = 0 \text{  if  } (Z < 0), \frac{\partial A}{\partial Z} = 1 \text{  if  } Z \geq 0 $


   



##### Tanh (Hyperbolic Tangent) Function:
 - $A = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \tanh(Z)$
 - $\frac{\partial A}{\partial Z} = 1 - \tanh^2(Z)$



##### Installation
To use the ActivationFunctions class, you can simply copy the class definition into your Python project. There are no additional dependencies required.

##### Usage
Instantiate the ActivationFunctions class, and then you can use its methods to compute activations and their derivatives.
```python
from activation_functions import ActivationFunctions
import numpy as np

# Example usage
Z = np.array([0.5, -1.2, 3.0, -0.8])

# Compute sigmoid activation
A_sigmoid, cache_sigmoid = ActivationFunctions.sigmoid(Z)
print("Sigmoid:", A_sigmoid)

# Compute ReLU activation
A_relu, cache_relu = ActivationFunctions.relu(Z)
print("ReLU:", A_relu)

# Compute tanh activation
A_tanh, cache_tanh = ActivationFunctions.tanh(Z)
print("Tanh:", A_tanh)
```
