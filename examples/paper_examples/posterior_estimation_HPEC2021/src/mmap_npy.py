"""
Description: MMAP NumPy handler 
Author      : Andrew Wood
Author_email: dewood@bu.edu
License     : Apache License, Version 2.0
"""





# SYSTEM IMPORTS
from typing import List, Tuple, Union
import numpy as np
import os
import sys


# PYTHON PROJECT IMPORTS


class MMapNpyHandle(object):
    def __init__(self,
                 path: str,
                 dtype: np.dtype,
                 shape: Tuple[int]) -> None:
        self.path: str = path
        self.dtype: np.dtype = dtype
        self.shape: Tuple[int] = shape

        if not self.path.endswith(".npy"):
            self.path += ".npy"

        self.data: np.ndarray = np.memmap(self.path, mode="w+", dtype=self.dtype,
                                          shape=self.shape, offset=128)
        self.header: str = np.lib.format.header_data_from_array_1_0(self.data)

    def __del__(self) -> None:
        del self.data
        with open(self.path, "r+b") as f:
            np.lib.format.write_array_header_1_0(f, self.header)

    def __enter__(self) -> "MMapNpyHandle":
        # open memmaped .npy file with offset of 128 so that we can write the .npy header later
        # based on: https://stackoverflow.com/questions/36769378/flushing-numpy-memmap-to-npy-file
        return self

    def __exit__(self, type, value, traceback) -> None:
        ...

    def write(self,
              idx: Union[int, List[int], np.ndarray],
              data: Union[int, float, np.ndarray]) -> None:
        self.data[idx] = data

    def contents(self) -> np.ndarray:
        return self.data


def open_mmap(path: str,
         dtype: np.dtype,
         shape: Tuple[int]) -> MMapNpyHandle:
    return MMapNpyHandle(path, dtype, shape)


# testing
if __name__ == "__main__":
    X: np.ndarray = np.random.randn(100, 100)

    with open_mmap("foo.npy", X.dtype, X.shape) as f:
        for i,r in enumerate(X):
            f.write(i, r)

    Y: np.ndarray = np.load("foo.npy")
    print(np.array_equal(X, Y))

