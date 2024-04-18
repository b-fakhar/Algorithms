## Logistic Regression Classifier
This Python class implements a logistic regression classifier. Logistic regression is a popular machine learning algorithm used for binary classification tasks.
- $Z = WX + b$
- $y_{pred} = \sigma(Z) = \frac{1}{1 + e^{-Z}}$
- $\text{Loss} = -\left(y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})\right)$
- $\text{Loss} = \frac{-1}{m} \sum\limits_{i=1}^{m} \left(y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})\right)$
- Gradient Descent:
  - $W := W - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}(y_{\text{pred}} - y_{\text{true}}) X$
  - $b := b - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}(y_{\text{pred}} - y_{\text{true}})$


###### - $m$ is the number of training examples.
###### - $X$ is the features.
###### - $y_{pred}$ is the predicted probability of the positive class.
###### - $y_{true}$ is the true label (0 or 1).
###### - $\alpha$ is the learning rate.

##### Installation

You can install the LogisticRegression class by cloning this repository or by directly copying the LogisticRegression class into your project.
##### Methods
- Fit the logistic regression model to the training data with optional early stopping and loss plotting.
  ```python
  fit(X_train, y_train, X_val=None, y_val=None, patience=None, print_loss=True, plot_loss=True):
  ```
- Predict class labels for test data.
  ```python
  predict(X_test):
  ```
##### Usage
```python
from logistic_regression import LogisticRegression
import numpy as np

# Create an instance of the LogisticRegression class
model = LogisticRegression()

# Fit the model to your training data
X_train = np.array([...])  # Your training features
y_train = np.array([...])  # Your training labels
model.fit(X_train, y_train)

# Predict labels for new data
X_test = np.array([...])  # Your test features
predictions = model.predict(X_test)

```
##### Dependencies
- NumPy
- Matplotlib (for plotting loss)
