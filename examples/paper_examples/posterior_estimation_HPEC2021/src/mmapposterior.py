"""
Description: Lazy MMAP Posterior class - use NVMe for the storage and DRAM for processing 
Author      : Andrew Wood
Author_email: dewood@bu.edu
License     : Apache License, Version 2.0
"""

# SYSTEM IMPORTS
from abc import ABC, abstractmethod
import numpy as np


# PYTHON PROJECT IMPORTS
from .posterior import Posterior
from .mmap_npy import open_mmap


class MMapPosterior(Posterior):
    def __init__(self,
                 num_params: int,
                 dtype: np.dtype = np.float32) -> None:
        super().__init__(num_params, dtype=dtype)

        # reserve space on the shelf for the mean and covariance
        self.posterior_sums = open_mmap(os.path.join(self.base_path,
                                                     "posterior_sums.npy"),
                                        self.dtype,
                                        (num_params, 1))
        self.posterior_squaresums = open_mmap(os.path.join(self.base_path,
                                                           "posterior_squaresums.npy"),
                                              self.dtype,
                                              (num_params, num_params)
        self.posterior_mu = open_mmap(os.path.join(self.base_path,
                                                     "posterior_mu.npy"),
                                       self.dtype,
                                      (num_params, 1))
        self.posterior_cov = open_mmap(os.path.join(self.base_path,
                                                    "posterior_cov.npy"),
                                       self.dtype,
                                       (num_params, num_params))

        self.posterior_sums.data *= 0
        self.posterior_squaresums.data *= 0
        self.posterior_mu.data *= 0
        self.poserior_cov.data *= 0


    def update(self,
               theta: np.ndarray) -> None:
        theta = theta.reshape(-1, 1)

        self.posterior_sums.data += theta
        self.posterior_squaresums.data += theta.dot(theta.T)

        # self.posterior_mu = self.posterior_sums /\
        #     (self.num_samples_seen + 1)

        # self.posterior_cov = (self.posterior_squaresums -
        #     self.posterior_mu.dot(self.posterior_sums.T) -
        #     self.posterior_sums.dot(self.posterior_mu.T) +
        #     (self.num_samples_seen + 1) * self.posterior_mu.dot(
        #         self.posterior_mu.T)) / max(self.num_samples_seen, 1)

        self.num_samples_seen += 1

    def finalize(self) -> None:
        # to generate samples, we need to perform the cholesky decomp
        # of our cov
        if(self.num_samples_seen < self.num_params):
            raise RuntimeError("ERROR: cov is singular, need %s more samples"
                               % (self.num_params-self.num_samples_seen))

        self.posterior_mu.data = self.posterior_sums.data / self.num_samples_seen

        self.posterior_cov.data = (self.posterior_squaresums.data -
            self.posterior_mu.data.dot(self.posterior_sums.data.T) -
            self.posterior_sums.data.dot(self.posterior_mu.data.T) +
            self.num_samples_seen * self.posterior_mu.data.dot(
                self.posterior_mu.data.T)) / (self.num_samples_seen - 1)

        # self.shelf.posterior_cholesky = np.linalg.cholesky(self.posterior_cov)

    def sample(self) -> np.ndarray:
        z: np.ndarray = np.random.normal(loc=0, scale=1,
                                         size=(self.num_params,1))

        return self.posterior_cholesky.dot(z)+self.posterior_mu.data
