"""
Description: Example of using mnist with lazy PyMM Posterior
Author      : Andrew Wood
Author_email: dewood@bu.edu
License     : Apache License, Version 2.0
"""

# SYSTEM IMPORTS
from typing import List
from tqdm import tqdm
import argparse
import csv
import numpy as np
import pymm
import os
import sys
import time
import torch as pt
import torch.nn.functional as F
import torchvision as ptv


_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from src.posterior import Posterior
from src.lazypymmposterior import LazyPymmPosterior
from mnist_models import Model_2Conv2FC, Model_2FC, Model_3FC, Model_4FC


model_map = {m.__name__.lower(): m for m in [Model_2Conv2FC, Model_2FC,
                                             Model_3FC, Model_4FC]}


def requires_flat_examples(m: pt.nn.Module) -> bool:
    result = False
    for model_type in [Model_2FC, Model_3FC, Model_4FC]:
        if isinstance(m, model_type):
            result = True
    return result


def train_one_epoch(m: pt.nn.Module,
                    optim: pt.optim.Optimizer,
                    loader: pt.utils.data.DataLoader,
                    epoch: int,
                    posterior: Posterior,
                    batch_posterior: int,
                    cuda: int) -> None:
    i = 1
    for batch_idx, (X, Y_gt) in tqdm(enumerate(loader),
                                     desc="training epoch %s" % epoch,
                                     total=len(loader)):
        optim.zero_grad()

        if requires_flat_examples(m):
            X = X.view(X.size(0), -1)
        if cuda > -1:
            X = X.to(cuda)

        Y_hat: pt.Tensor = m.forward(X)
        loss: pt.Tensor = F.nll_loss(Y_hat.cpu(), Y_gt.cpu())
        loss.backward()
        optim.step()

        if ( i%batch_posterior == 0):
            posterior.update(m.get_params())
        i += 1


def main() -> None:
    script_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=sorted(model_map.keys()),
                        help="the model type")
    parser.add_argument("-K", "--posterior_sample_size", type=int,
                        default=np.inf, help="the number of samples posterior expects")
    parser.add_argument("-b", "--batch_size", type=int, default=100,
                        help="batch size")
    parser.add_argument("-n", "--learning_rate", type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0,
                        help="sgd momentum")
    parser.add_argument("-e", "--epochs", type=int, default=int(1e6),
                        help="num epochs")
    parser.add_argument("-p", "--path", type=str,
                        default="/scratch/aewood/data/mnist")
    parser.add_argument("-c", "--cuda", type=int,
                        default=-1,
                        help="gpu id (-1 for cpu)")
    parser.add_argument("-s", "--size_mb", type=int, default=40000,
                        help="size of shelf in mb")
    parser.add_argument("-f", "--shelf_file", type=str,
                        default="/mnt/pmem0",
                        help="pymm shelf directory")
    parser.add_argument("-r", "--bpost", type=int,
                        default=1)
    parser.add_argument("-back", "--backend", type=str,
                        default="hstore-cc", choices=["hstore-cc", "mapstore"],
                        help="The backend of the shelf - hstore-cc/ mapstore")
    parser.add_argument("-csv", "--results_filepath", type=str,
                        default="./results/mnist/pymm_timings.csv")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    results_dir: str = os.path.abspath(os.path.dirname(args.results_filepath))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    start_time: float = time.time()
    train_loader = pt.utils.data.DataLoader(
        ptv.datasets.MNIST(args.path, train=True, download=True,
                           transform=ptv.transforms.ToTensor()
                            # transform=ptv.transforms.Compose([
                            #     ptv.transforms.ToTensor(),
                            #     ptv.transforms.Normalize((0.1307,),
                            #                              (0.3081,))
                            # ])
                           ),
        batch_size=args.batch_size,
        shuffle=True)
    end_train_loader_time = time.time() - start_time

    start_time = time.time()
    test_loader = pt.utils.data.DataLoader(
        ptv.datasets.MNIST(args.path, train=False, download=True,
                           transform=ptv.transforms.ToTensor()
                            # transform=ptv.transforms.Compose([
                            #     ptv.transforms.ToTensor(),
                            #     ptv.transforms.Normalize((0.1307,),
                            #                              (0.3081,))
                            # ])
                           ),
        batch_size=args.batch_size,
        shuffle=True)
    end_test_loader_time = time.time() - start_time

    if np.isinf(args.posterior_sample_size):
        args.posterior_sample_size = len(train_loader) * args.epochs

    start_time = time.time()
    shelf = pymm.shelf("mnist_pymm_posterior",
                       size_mb=args.size_mb,
                       pmem_path=args.shelf_file,
                       backend=args.backend,
                       force_new=True)
    end_shelf_time = time.time() - start_time

    print("loading model")

    start_time = time.time()
    m = model_map[args.model]()
    if args.cuda > -1:
        m = m.to(args.cuda)
    optimizer = pt.optim.SGD(m.parameters(), lr=args.learning_rate,
                             momentum=args.momentum)
    end_make_model_time = time.time() - start_time

    num_params: int = m.get_params().shape[0]

    start_time = time.time()
    posterior = LazyPymmPosterior(num_params, shelf, K=args.posterior_sample_size)
    end_make_posterior_time = time.time() - start_time
    print("num params: %s" % num_params,
          "posterior will take %s bytes" % posterior.nbytes)

    epoch_times: List[float] = list()
    start_experiment_time = time.time()
    # update posterior with first parameter sample
    posterior.update(m.get_params())
    for e in range(args.epochs):
        start_epoch_time = time.time()
        train_one_epoch(m,
                        optimizer,
                        train_loader,
                        e+1,
                        posterior,
                        args.bpost,
                        args.cuda)
        epoch_times.append(time.time() - start_epoch_time)

    print("finalizing posterior")
    start_finalize_time = time.time()
    posterior.finalize()
    end_finalize_time = time.time() - start_finalize_time
    print("done")

    end_experiment_time = time.time() - start_experiment_time
    end_script_time = time.time() - script_start_time
    # posterior.sample()

    with open(args.results_filepath, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["posterior size (byte)", posterior.nbytes])
        writer.writerow(["train data loading time (s)", end_train_loader_time])
        writer.writerow(["test data loading time (s)", end_test_loader_time])
        writer.writerow(["open shelf time (s)", end_shelf_time])
        writer.writerow(["make model time (s)", end_make_model_time])
        writer.writerow(["make posterior time (s)", end_make_posterior_time])
        for e,t in enumerate(epoch_times):
            writer.writerow(["epoch %s time (s)" % e, t])
        writer.writerow(["posterior finalize time (s)", end_finalize_time])
        writer.writerow(["total experiment time (s)", end_experiment_time])
        writer.writerow(["total script time (s)", end_script_time])


if __name__ == "__main__":
    main()

