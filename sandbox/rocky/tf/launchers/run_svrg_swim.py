#!/usr/bin/env python
"""
Tianbing Xu 7-10-2017
"""
import multiprocessing
import sys
from subprocess import call
import numpy as np

prefix_map = {
    'c': 'cartpole_swing_up',
    'd': 'pendulum',
    'car': 'cartpole',
    'mou': 'mountain_car',
    'swim': 'swimmer',
    'hopper': 'hopper',
    'walker': 'walker',
    'cheetah': 'cheetah',
    'humanoid': 'humanoid',
}

algorithm_map = {
    'tr': 'trpo',
    'svrg': 'svrg',
}

root_dir = "../logs/log_svrg_swim"
seed = 1
game = 'swim'
algorithm = "svrg"
batch_size = 50000
mini_batch_sizes = [5000]
max_path_length = 1000
delta = 0.01
n_itr = 500
max_epochs = 1
cg_iters = 10
subsample_factor = 0.1
max_batchs = [20]


if __name__ == "__main__":
    for mini_batch_size in mini_batch_sizes:
        for max_batch in max_batchs:
            command = "python benchmark_svrg.py {:} {:}  {:} {:} {:} {:} {:} {:} {:} {:} {:} {:} {:}".format(
                root_dir,
                algorithm,
                game,
                seed,
                batch_size,
                mini_batch_size,
                n_itr,
                max_path_length,
                delta,
                max_epochs,
                cg_iters,
                subsample_factor,
                max_batch)
            print(command)
            call(command, shell=True)

            plotOut = "fig.{:}_{:}_{:}_{:}_{:}.png".format(
                algorithm, game, n_itr, mini_batch_size, max_batch)
            plotcmd = "python plotComparison.py -o {:} -a {:} -e {:} -t {:} -m 1 -p 0 -i {:}".format(
                plotOut, algorithm_map[algorithm], prefix_map[game], n_itr, root_dir)
            print(plotcmd)
    #call(plotcmd, shell=True)
