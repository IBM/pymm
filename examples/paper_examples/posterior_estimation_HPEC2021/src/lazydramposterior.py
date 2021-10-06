"""
Description: Lazy DRAM Posterior class 
Author      : Andrew Wood
Author_email: dewood@bu.edu
License     : Apache License, Version 2.0
"""





# SYSTEM IMPORTS
from abc import ABC, abstractmethod
import numpy as np


# PYTHON PROJECT IMPORTS
from .posterior import Posterior


class LazyDramPosterior(Posterior):
    def __init__(self,
                 num_params: int,
                 K: int = 15,
                 dtype: np.dtype = np.float32) -> None:
        super().__init__(num_params, dtype=dtype)
        self.K: int = K

        self.mu: np.ndarray = np.zeros((self.num_params, 1), dtype=self.dtype)
        self.sec_moment_uncentered: np.ndarray = np.zeros((self.num_params, 1),
                                                          dtype=self.dtype)

        self.diag: np.ndarray = np.zeros((self.num_params, 1), dtype=self.dtype)
        # self.cov: np.ndarray = np.zeros((self.num_params, self.num_params),
        #                                 dtype=self.dtype)
        self.D_hat: np.ndarray = np.zeros((self.num_params, K), dtype=self.dtype)
        self.D_hat_idx: int = 0

        self.num_samples: int = 0

    def update(self,
               theta: np.ndarray) -> None:
        theta = theta.reshape(-1,1)
        self.num_samples += 1
        self.mu += theta

        self.sec_moment_uncentered += theta**2

        self.D_hat[:,self.D_hat_idx] = (theta-(self.sec_moment_uncentered/self.num_samples)).reshape(-1)
        self.D_hat_idx = (self.D_hat_idx + 1) % self.K

    def finalize(self) -> None:
        self.mu /= self.num_samples
        self.diag = self.sec_moment_uncentered - (self.mu**2)
        # self.cov = (self.D_hat @ self.D_hat.T) / (self.K - 1)

    @property
    def nbytes(self) -> int:
        return self.mu.nbytes + self.sec_moment_uncentered.nbytes +\
               self.diag.nbytes + self.D_hat.nbytes + 4 + 4 + 4

    def sample(self) -> np.ndarray:
        z1: np.ndarray = np.random.normal(loc=0, scale=1,
                                          size=(self.num_params,1))
        z2: np.ndarray = np.random.normal(loc=0, scale=1,
                                          size=(self.K, 1))

        return self.mu + (1 / np.sqrt(2)) * np.sqrt(self.diag) * z1 +\
               (1 / (np.sqrt(2 * (self.K-1)))) * self.D_hat.dot(z2)
