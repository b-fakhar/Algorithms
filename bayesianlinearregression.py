"""
Implementation of Bayesian Linear Regression.

Programmed by Behnaaz Fakhar <fakhar.behnaz@gmail.com>

*    2019-04-25 Initial coding
"""

import numpy as np

class BayesianLinearRegression:
    def __init__(self, prior_covariance: np.ndarray, prior_mean: np.ndarray, noise_variance: float):
        """
        Initializes the Bayesian Linear Regression model.

        Args:
            prior_covariance (np.ndarray): Prior covariance matrix.
            prior_mean (np.ndarray): Prior mean vector.
            noise_variance (float): Noise variance.
        """
        if not isinstance(prior_covariance, np.ndarray):
            raise TypeError("prior_covariance must be a numpy array.")
        if not isinstance(prior_mean, np.ndarray):
            raise TypeError("prior_mean must be a numpy array.")
        if not isinstance(noise_variance, (int, float)):
            raise TypeError("noise_variance must be a number.")
        if prior_covariance.ndim != 2 or prior_mean.ndim != 1:
            raise ValueError("prior_covariance must be a 2D array and prior_mean must be a 1D array.")

        self.prior_covariance = prior_covariance
        self.prior_mean = prior_mean
        self.noise_variance = noise_variance
        self.posterior_mean = prior_mean

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target variable for given input features.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted target variable.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return np.dot(X, self.posterior_mean)

    def update(self, X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Updates the model using new data.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target variable.

        Returns:
            np.ndarray: Updated posterior mean.
            np.ndarray: Updated posterior covariance.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be a 2D array and y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be equal.")

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        noise_variance_inv = 1.0 / self.noise_variance
        prior_covariance_inv = np.linalg.inv(self.prior_covariance)

        # Compute posterior covariance
        self.posterior_covariance = np.linalg.inv(noise_variance_inv * np.dot(X.T, X) + prior_covariance_inv)

        # Compute posterior mean
        self.posterior_mean = np.dot(self.posterior_covariance,
                                     noise_variance_inv * np.dot(X.T, y) + np.dot(prior_covariance_inv, self.prior_mean))

        return self.posterior_mean, self.posterior_covariance
