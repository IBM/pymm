"""
Description: PyMM Posterior class - use persistent memory for storage and DRAM processing
Author      : Andrew Wood
Author_email: dewood@bu.edu
License     : Apache License, Version 2.0
"""






# SYSTEM IMPORTS
from abc import ABC, abstractmethod
import numpy as np
import pymm


# PYTHON PROJECT IMPORTS
from .posterior import Posterior


class PymmPosterior(Posterior):
    def __init__(self,
                 num_params: int,
                 shelf,
                 dtype: np.dtype = np.float32) -> None:
        super().__init__(num_params, dtype=dtype)

        self.shelf = shelf

        # reserve space on the shelf for the mean and covariance
        self.shelf.posterior_sums = pymm.ndarray((num_params, 1),
                                               dtype=self.dtype)
        self.shelf.posterior_squaresums = pymm.ndarray((num_params,
                                                        num_params),
                                                        dtype=self.dtype)
        self.shelf.posterior_mu = pymm.ndarray((num_params, 1),
                                               dtype=self.dtype)
        self.shelf.posterior_cov = pymm.ndarray((num_params, num_params),
                                                dtype=self.dtype)

        # i know that creating an array defaults the values to 0,
        # but i want to make sure that this is always the case
        self.shelf.posterior_sums.fill(0)
        self.shelf.posterior_squaresums.fill(0)
        self.shelf.posterior_mu.fill(0)
        self.shelf.posterior_cov.fill(0)


    def update(self,
               theta: np.ndarray) -> None:
        theta = theta.reshape(-1, 1)

        self.shelf.posterior_sums += theta
        self.shelf.posterior_squaresums += theta.dot(theta.T)

        # self.shelf.posterior_mu = self.shelf.posterior_sums /\
        #     (self.num_samples_seen + 1)

        # self.shelf.posterior_cov = (self.shelf.posterior_squaresums -
        #     self.shelf.posterior_mu.dot(self.shelf.posterior_sums.T) -
        #     self.shelf.posterior_sums.dot(self.shelf.posterior_mu.T) +
        #     (self.num_samples_seen + 1) * self.shelf.posterior_mu.dot(
        #         self.shelf.posterior_mu.T)) / max(self.num_samples_seen, 1)

        self.num_samples_seen += 1

    def finalize(self) -> None:
        # to generate samples, we need to perform the cholesky decomp
        # of our cov
        if(self.num_samples_seen < self.num_params):
            raise RuntimeError("ERROR: cov is singular, need %s more samples"
                               % (self.num_params-self.num_samples_seen))

        self.shelf.posterior_mu = self.shelf.posterior_sums /\
            self.num_samples_seen

        self.shelf.posterior_cov = (self.shelf.posterior_squaresums -
            self.shelf.posterior_mu.dot(self.shelf.posterior_sums.T) -
            self.shelf.posterior_sums.dot(self.shelf.posterior_mu.T) +
            (self.num_samples_seen) * self.shelf.posterior_mu.dot(
                self.shelf.posterior_mu.T)) / (self.num_samples_seen - 1)

        # self.shelf.posterior_cholesky = np.linalg.cholesky(self.shelf.posterior_cov)

    def sample(self) -> np.ndarray:
        z: np.ndarray = np.random.normal(loc=0, scale=1,
                                         size=(self.num_params,1))

        return self.shelf.posterior_cholesky.dot(z)+self.shelf.posterior_mu

