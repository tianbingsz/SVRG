#!/usr/bin/python
"""
Tianbing Xu, 4/28/2017
reference and borrow ideas and code from Yang Liu from UIUC
"""
import sys
import matplotlib
# the following line is added immediately after import matplotlib
# and before import pylot. The purpose is to ensure the plotting
# works even under remote login (i.e. headless display)
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as pyplot
from csv import reader
import numpy as np
import argparse
import re
import os
import pdb

"""Plot Average Performance from progress.csv.
usage: Plot average performance from different progress.csv.
       [-h] [-p PLOTOPTION] [-k KEY] [-i INPUT_DIR]
       [-f [INPUT_FILES [INPUT_FILES ...]]] [-t ITERATION]
       [-m [MEMS [MEMS ...]]] [-o OUTPUT] [-a ALGORITHM] [-e ENV]
       [--format FORMAT]

optional arguments:
  -h, --help            show this help message and exit
  -p PLOTOPTION, --plotOption PLOTOPTION
                        plot options: return-iteration(1), return-sample(2),
                        ess(3)
  -k KEY, --key KEY     key of scores to plot, the default is AverageReturn
  -i INPUT_DIR, --input_dir INPUT_DIR
                        input dir of progress.csv default will be ../logs/
  -f [INPUT_FILES [INPUT_FILES ...]], --input_files [INPUT_FILES [INPUT_FILES ...]]
                        input files of progress.csv
  -t ITERATION, --iteration ITERATION
                        num of iterations, default 100
  -m [MEMS [MEMS ...]], --mems [MEMS [MEMS ...]]
                        different memory cap
  -o OUTPUT, --output OUTPUT
                        output filename of figure, default will be standard
                        output
  -a ALGORITHM, --algorithm ALGORITHM
                        agent algorithm, default will be is
  -e ENV, --env ENV     environment, default will be cartpole
  --format FORMAT       figure format(png|pdf|ps|eps|svg)
plot return vs iterations
python plotComparison.py -o fig.5.2.png -a importance_sampler_reinforce -e
cartpole -t 100 -m 1 2 -p 1

plot return vs samples
python plotComparison.py -o fig.5.2.png -a importance_sampler_reinforce -e
cartpole -t 100 -m 1 2 -p 2

plot ess
python plotComparison.py -o fig.5.2.png -a importance_sampler_reinforce -e cartpole -f
importance_sampler_reinforce_seed=2_iter=100_mem=10_env=cartpole_2017-05-01_14:30:13.520344
importance_sampler_reinforce_seed=2_iter=100_mem=10_env=cartpole_2017-05-01_14:37:48.242995
-t 100 -m 1 -p 3
"""


def load_average_return(filename, label='AverageReturn', n_itr=100):
    with open(filename, 'r') as f:
        data = list(reader(f))
    if data is None or len(data) < n_itr + 1:
        return None
    rIdx = data[0].index(label)
    returns = [d[rIdx] for d in data[1::]]
    return returns


def get_num_samples(filename, label='NumSamples', n_itr=100):
    with open(filename, 'r') as f:
        data = list(reader(f))
    if data is None or len(data) < n_itr + 1:
        return None
    rIdx = data[0].index(label)
    num_samples = [d[rIdx] for d in data[1::]]
    return num_samples


def get_file_list(root_dir="../logs/", n_itr=100, **kwargs):
    file_list = os.listdir(root_dir)
    # importance_sampler_seed=2_iter=10_mem=1_env=cartpole_2017-04-28_17
    # algo_seed=1_iter=10_mem=1_env=env_day_sec
    file_list = [f for f in file_list
                 if str(n_itr) == f.split("iter")[1].split("_")[0].split('=')[1]]
    if 'algo' in kwargs:
        file_list = [f for f in file_list
                     if kwargs['algo'] == f.split('seed')[0][:-1]]
    if 'env' in kwargs:
        file_list = [f for f in file_list
                     if kwargs['env'] == f.split('_')[:-2][-2].split('=')[-1]]
    if 'mem' in kwargs:
        file_list = [f for f in file_list
                     if str(kwargs['mem']) == f.split('_')[:-5][-1].split('=')[1]]
    for f in file_list:
        print("{:}".format(f))
    file_list = [os.path.join(root_dir, f) for f in file_list]
    return file_list


def get_multi_performance(root_dir="../logs/", label="AverageReturn",
                          n_itr=100, **kwargs):
    file_list = get_file_list(root_dir, n_itr, **kwargs)
    perf = [load_average_return(os.path.join(filename, "progress.csv"),
                                label=label, n_itr=n_itr) for filename in file_list]
    perf = [x for x in perf if x is not None]
    perf = np.array(perf).astype(np.float)
    perf_mean = np.mean(perf, axis=0)
    perf_std = np.std(perf, axis=0)
    num_samples = get_num_samples(os.path.join(file_list[0], "progress.csv"),
                                  label='NumSamples', n_itr=100)
    return perf_mean, perf_std, np.array(num_samples).astype(np.int)


def get_performance(in_file="", label="ESS of all weights",
                    n_itr=100, **kwargs):
    perf = load_average_return(os.path.join(in_file, "progress.csv"),
                               label=label, n_itr=n_itr)
    return perf


"""
plot the effective sample size for a multiple files
"""


def plot_ess(algorithm=None, in_files=[], env=None,
             label="ESS of all weights", n_itr=100, outputfile=None):
    for idx, in_file in enumerate(in_files):
        performance = get_performance(algo=algorithm, env=env,
                                      label=label, in_file=in_file, n_itr=n_itr)
        pyplot.plot(range(len(performance)), performance, linewidth=3.0,
                    label="ESS" + str(idx))

    pyplot.title(label + " over iterations")
    pyplot.xlabel("iterations")
    pyplot.ylabel(label)
    pyplot.yscale('log')
    pyplot.legend()
    pyplot.savefig(outputfile, bbox_inches='tight')


"""
run mulitple times of the algorithm and plot the average performance
"""


def plot_average_performance(algorithm=None, root_dir="../logs/",
                             env=None, label="AverageReturn", n_itr=100, outputfile=None):
    perf_mean, perf_std, _ = get_multi_performance(algo=algorithm, env=env,
                                                   label=label, root_dir=root_dir, n_itr=n_itr)
    pyplot.plot(range(len(perf_mean)), perf_mean, linewidth=3.0)
    pyplot.title(label + " over iterations")
    pyplot.xlabel("iterations")
    pyplot.ylabel(label)
    pyplot.fill_between(range(len(perf_mean)), perf_mean - perf_std,
                        perf_mean + perf_std, alpha=0.3)
    # pyplot.legend()
    pyplot.savefig(outputfile, bbox_inches='tight')


"""
run mulitple times of experiments with different mem
and plot the average performance comparison
"""


def plot_mem_performance_comparsion(algorithm=None, root_dir="../logs/",
                                    env=None, label="AverageReturn",
                                    n_itr=100, mems=[], outputfile=None):
    for mem in mems:
        perf_mean, perf_std, _ = get_multi_performance(algo=algorithm, env=env,
                                                       label=label, root_dir=root_dir,
                                                       n_itr=n_itr, mem=mem)
        pyplot.plot(
            range(len(perf_mean)),
            perf_mean,
            linewidth=3.0,
            label=str(mem))
        pyplot.fill_between(range(len(perf_mean)), perf_mean - perf_std,
                            perf_mean + perf_std, alpha=0.3)

    pyplot.xlabel("iterations")
    pyplot.ylabel(label)
    pyplot.title(label + " over iterations for different mem")
    pyplot.legend()
    pyplot.savefig(outputfile, bbox_inches='tight')


"""
run mulitple times of experiments with different mem
and plot the average performance comparison w/ num_samples
"""


def plot_mem_performance_comparsion_samples(algorithm=None, root_dir="../logs/",
                                            env=None, label="AverageReturn",
                                            n_itr=100, mems=[], outputfile=None):
    for mem in mems:
        perf_mean, perf_std, num_samples = get_multi_performance(
            algo=algorithm, env=env,
            label=label, root_dir=root_dir,
            n_itr=n_itr, mem=mem)
        pyplot.plot(
            num_samples,
            perf_mean,
            linewidth=3.0,
            label=str(mem))
        pyplot.fill_between(num_samples, perf_mean - perf_std,
                            perf_mean + perf_std, alpha=0.3)

    pyplot.xlabel("samples")
    pyplot.xscale('log')
    pyplot.ylabel(label)
    pyplot.title(label + " over samples for different mem")
    pyplot.legend()
    pyplot.savefig(outputfile, bbox_inches='tight')


def main(argv):
    """
    plotting curves for average performance.
    """
    cmdparser = argparse.ArgumentParser(
        "Plot average performance from different progress.csv.")
    cmdparser.add_argument(
        '-p',
        '--plotOption',
        help='plot options: return-iteration(1), return-sample(2), ess(3)')
    cmdparser.add_argument(
        '-k',
        '--key',
        help='key of scores to plot, the default is AverageReturn')
    cmdparser.add_argument(
        '-i',
        '--input_dir',
        help='input dir of progress.csv '
        'default will be ../logs/')
    cmdparser.add_argument(
        '-f',
        '--input_files',
        nargs='*',
        help='input files of progress.csv ')
    cmdparser.add_argument(
        '-t',
        '--iteration',
        help='num of iterations, default 100')
    cmdparser.add_argument(
        '-m',
        '--mems',
        nargs='*',
        help='different memory cap')
    cmdparser.add_argument(
        '-o',
        '--output',
        help='output filename of figure, '
        'default will be standard output')
    cmdparser.add_argument(
        '-a',
        '--algorithm',
        help='agent algorithm, default will be is')
    cmdparser.add_argument(
        '-e',
        '--env',
        help='environment, default will be cartpole')
    cmdparser.add_argument(
        '--format',
        help='figure format(png|pdf|ps|eps|svg)')
    args = cmdparser.parse_args(argv)
    format = args.format
    if args.output:
        outputfile = open(args.output, 'wb')
        if not format:
            format = os.path.splitext(args.output)[1]
            if not format:
                format = 'png'
    else:
        outputfile = sys.stdout

    root_dir = args.input_dir
    if root_dir is None:
        root_dir = "../logs/"
    algorithm = args.algorithm
    if algorithm is None:
        algorithm = "is"
    env = args.env
    if env is None:
        env = "cartpole"
    label = args.key
    if label is None:
        label = "AverageReturn"
    n_itr = int(args.iteration)
    if n_itr is None:
        n_itr = 100

    mems = [int(i) for i in args.mems]
    print("root_dir : {:}, algorithm: {:}, env : {:}, label: {:}".format(
        root_dir, algorithm, env, label))
    if args.plotOption == '1':
        plot_mem_performance_comparsion(
            root_dir=root_dir,
            algorithm=algorithm,
            env=env,
            label=label,
            n_itr=n_itr,
            mems=mems,
            outputfile=outputfile)
    if args.plotOption == '2':
        plot_mem_performance_comparsion_samples(
            root_dir=root_dir,
            algorithm=algorithm,
            env=env,
            label=label,
            n_itr=n_itr,
            mems=mems,
            outputfile=outputfile)

    in_files = []
    if args.input_files is not None:
        for input_file in args.input_files:
            in_files.append(root_dir + input_file)

    if args.plotOption == '3':
        plot_ess(
            in_files=in_files,
            algorithm=algorithm,
            env=env,
            label="ESS of all weights",
            n_itr=n_itr,
            outputfile=outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
