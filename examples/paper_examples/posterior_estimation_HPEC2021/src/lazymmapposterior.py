"""
Description: Lazy MMAP Posterior class - use NVMe for the storage and DRAM for processing 
Author      : Andrew Wood
Author_email: dewood@bu.edu
License     : Apache License, Version 2.0
"""





# SYSTEM IMPORTS
from abc import ABC, abstractmethod
import numpy as np
import os


# PYTHON PROJECT IMPORTS
from .posterior import Posterior
from .mmap_npy import open_mmap


class LazyMMapPosterior(Posterior):
    def __init__(self,
                 num_params: int,
                 posterior_path: str,
                 K: int = 15,
                 dtype: np.dtype = np.float32) -> None:
        super().__init__(num_params, dtype=dtype)
        self.K: int = K

        mu_path: str = os.path.join(posterior_path, "mu.npy")
        sec_moment_uncentered_path: str = os.path.join(posterior_path,
                                                       "sec_moment_uncentered.npy")
        diag_path: str = os.path.join(posterior_path, "cov_diag.npy")
        D_hat_path: str = os.path.join(posterior_path, "D_hat.npy")

        self.mu: np.ndarray = open_mmap(path=mu_path,
                                        dtype=self.dtype,
                                        shape=(self.num_params, 1))
        self.sec_moment_uncentered: np.ndarray = open_mmap(path=sec_moment_uncentered_path,
                                                           dtype=self.dtype,
                                                           shape=(self.num_params, 1))

        self.diag: np.ndarray = open_mmap(path=diag_path,
                                          dtype=self.dtype,
                                          shape=(self.num_params, 1))
        # self.cov: np.ndarray = np.zeros((self.num_params, self.num_params),
        #                                 dtype=self.dtype)
        self.D_hat: np.ndarray = open_mmap(path=D_hat_path,
                                           dtype=self.dtype,
                                           shape=(self.num_params, K))
        self.D_hat_idx: int = 0

        self.num_samples: int = 0

    def update(self,
               theta: np.ndarray) -> None:
        theta = theta.reshape(-1,1)
        self.num_samples += 1
        self.mu.data += theta

        self.sec_moment_uncentered.data += theta**2

        self.D_hat.data[:,self.D_hat_idx] = (theta-(self.sec_moment_uncentered.data/self.num_samples)).reshape(-1)
        self.D_hat_idx = (self.D_hat_idx + 1) % self.K

    def finalize(self) -> None:
        self.mu.data /= self.num_samples
        self.diag.data = self.sec_moment_uncentered.data - (self.mu.data**2)
        # self.cov = (self.D_hat @ self.D_hat.T) / (self.K - 1)

    @property
    def nbytes(self) -> int:
        return self.mu.data.nbytes + self.sec_moment_uncentered.data.nbytes +\
               self.diag.data.nbytes + self.D_hat.data.nbytes + 4 + 4 + 4

    def sample(self) -> np.ndarray:
        z1: np.ndarray = np.random.normal(loc=0, scale=1,
                                          size=(self.num_params,1))
        z2: np.ndarray = np.random.normal(loc=0, scale=1,
                                          size=(self.K, 1))

        return self.mu.data + (1 / np.sqrt(2)) * np.sqrt(self.diag.data) * z1 +\
               (1 / (np.sqrt(2 * (self.K-1)))) * self.D_hat.data.dot(z2)
