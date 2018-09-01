#!/usr/bin/python

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

"""Plot AverageReturn from progress.csv.
To use this script to generate plot for AverageReturn:
    python plotcurve.py -i progress.csv -o fig.png
    python plotcurve.py -i progress.csv -o fig.png AverageReturn

usage: [-h] [-i INPUT] [-o OUTPUT] [--format FORMAT] [key [key ...]]

positional arguments:
  key                   keys of scores to plot, the default will be
                        AverageReturn

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input filename of log, default will be standard
                        input
  -o OUTPUT, --output OUTPUT
                        output filename of figure, default will be standard
                        output
  --format FORMAT       figure format(png|pdf|ps|eps|svg)


"""


def plot_average_return(keys, inputfile, outputfile):
    print("open file ", inputfile)
    # pdb.set_trace()
    with open(inputfile, 'r') as f:
        data = list(reader(f))
    # data[0] header
    # ['AverageDiscountedReturn', 'MinReturn', 'Entropy', 'StdReturn',
    # 'NumTrajs', 'PolicyExecTime', 'MaxKL', 'EnvExecTime', 'Time',
    # 'Perplexity', 'AverageReturn', 'AveragePolicyStd', 'LossAfter',
    # 'LossBefore', 'ExplainedVariance', 'MeanKL', 'MaxReturn',
    # 'ProcessExecTime', 'Iteration', 'ItrTime']
    # data[1::] recorded statistics
    if len(data) < 2:
        return

    if not keys:
        key = 'AverageReturn'
    else:
        key = keys[0]
    rIdx = data[0].index(key)
    returns = [d[rIdx] for d in data[1::]]
    # plot
    pyplot.plot(range(len(returns)), returns, linewidth=3.0)
    pyplot.title(key + ' ' + ' over iterations')
    # pyplot.xscale('log')
    pyplot.xlabel('iteration')
    pyplot.ylabel(key)
    # pyplot.show()
    pyplot.savefig(outputfile, bbox_inches='tight')
    pyplot.clf()
    print("save to output file")


def removeDuplicates(num_samples):
    num_samples.append(0)
    unique_idx = [i for i in range(len(num_samples) - 1) if num_samples[i] !=
                  num_samples[i + 1]]
    return unique_idx


def plot_average_return_sample(keys, inputfile, outputfile):
    print("open file ", inputfile)
    # pdb.set_trace()
    with open(inputfile, 'r') as f:
        data = list(reader(f))
    # data[0] header
    # [NumSamples,NumTrajs,MaxKL,Iteration,TotalRho,AverageDiscountedReturn,
    #  MinReturn,TotalWeights,LossAfter,Time,AverageReturn,AveragePolicyStd,
    #  MaxReturn,StdReturn,ESS,ItrTime,TotalLoglik,LossBefore,Entropy,MeanKL,Perplexity]
    # data[1::] recorded statistics
    if len(data) < 2:
        return

    if not keys:
        key = 'AverageReturn'
    else:
        key = keys[0]
    rIdx = data[0].index(key)
    returns = [d[rIdx] for d in data[1::]]
    num_samples = [d[data[0].index('NumSamples')] for d in data[1::]]
    # samples [5000, 5000, ....] returns[-230, -230, ....]
    idx = removeDuplicates(num_samples)
    num_samples = np.array(num_samples)
    num_samples = num_samples[idx]
    returns = np.array(returns)
    returns = returns[idx]
    # plot
    pyplot.plot(num_samples, returns, linewidth=3.0)
    pyplot.title(key + ' ' + ' over samples')
    pyplot.xlabel('samples')
    pyplot.ylabel(key)
    # pyplot.show()
    pyplot.savefig(outputfile, bbox_inches='tight')
    pyplot.clf()
    print("save to output file")


def plot_multiple_average_return_sample(keys, inputfiles, outputfile):
    # pdb.set_trace()

    data_list = []
    labels = ['svrg', 'stein_trpo']
    for inputfile in inputfiles:
        print("open file ", inputfile)
        with open(inputfile, 'r') as f:
            data_list.append(list(reader(f)))
    for idx, data in enumerate(data_list):
        # data[0] header
        # [NumSamples,NumTrajs,MaxKL,Iteration,TotalRho,AverageDiscountedReturn,
        #  MinReturn,TotalWeights,LossAfter,Time,AverageReturn,AveragePolicyStd,
        #  MaxReturn,StdReturn,ESS,ItrTime,TotalLoglik,LossBefore,Entropy,MeanKL,Perplexity]
        # data[1::] recorded statistics
        if len(data) < 2:
            return

        if not keys:
            key = 'AverageReturn'
        else:
            key = keys[0]
        rIdx = data[0].index(key)
        returns = [d[rIdx] for d in data[1::]]
        #num_samples = [d[data[0].index('NumSamples')] for d in data[1::]]
        # samples [5000, 5000, ....] returns[-230, -230, ....]
        #idx = removeDuplicates(num_samples)
        #num_samples = np.array(num_samples)
        #num_samples = num_samples[idx]
        #returns = np.array(returns)
        #returns = returns[idx]
        # plot
        pyplot.plot(range(len(returns)), returns,
                    linewidth=3.0, label=labels[idx])
        # pyplot.show()
    pyplot.title(key + ' ' + ' over samples')
    #pyplot.xscale('log')
    pyplot.xlabel('samples')
    pyplot.ylabel(key)
    pyplot.savefig(outputfile, bbox_inches='tight')
    pyplot.clf()
    print("save to output file")


def plot_multiple_average_return_time(keys, inputfiles, outputfile):
    # pdb.set_trace()

    data_list = []
    for inputfile in inputfiles:
        print("open file ", inputfile)
        with open(inputfile, 'r') as f:
            data_list.append(list(reader(f)))
    labels = ['stein_trpo']
    for idx, data in enumerate(data_list):
        # data[0] header
        # [NumSamples,NumTrajs,MaxKL,Iteration,TotalRho,AverageDiscountedReturn,
        #  MinReturn,TotalWeights,LossAfter,Time,AverageReturn,AveragePolicyStd,
        #  MaxReturn,StdReturn,ESS,ItrTime,TotalLoglik,LossBefore,Entropy,MeanKL,Perplexity]
        # data[1::] recorded statistics
        if len(data) < 2:
            return

        if not keys:
            key = 'Average Returns'
        else:
            key = keys[0]
        rIdx = data[0].index('AverageReturn')
        returns = [d[rIdx] for d in data[1::]]
        #num_samples = [d[data[0].index('NumSamples')] for d in data[1::]]
        #times = [d[data[0].index('ItrTime')] for d in data[1::]]
        # samples [5000, 5000, ....] returns[-230, -230, ....]
        #idx = removeDuplicates(num_samples)
        #times = np.array(times)
        #times = times[idx]
        #returns = np.array(returns)
        #returns = returns[idx]
        # plot
        # pdb.set_trace()
        #pyplot.plot(np.cumsum(times), returns, linewidth=3.0)
        pyplot.plot(range(len(returns)), returns,
                    linewidth=3.0, label=labels[idx])
        # pyplot.show()
    pyplot.title(key + ' ' + ' over iterations', fontsize=16)
    pyplot.xlabel('iteration',fontsize=16)
    pyplot.ylabel(key, fontsize=16)
    pyplot.tick_params(axis='x', labelsize=16)
    pyplot.tick_params(axis='y', labelsize=16)
    pyplot.legend()
    pyplot.savefig(outputfile, bbox_inches='tight')
    pyplot.clf()
    print("save to output file")


def main(argv):
    """
    main method of plotting curves.
    """
    cmdparser = argparse.ArgumentParser(
        "Plot AverageReturn from progress.csv.")
    cmdparser.add_argument(
        'key', nargs='*',
        help='key of scores to plot, the default is AverageReturn')
    cmdparser.add_argument(
        '-i',
        '--inputs',
        nargs='*',
        help='input filename(s) of progress.csv '
        'default will be standard input')
    cmdparser.add_argument(
        '-o',
        '--output',
        help='output filename of figure, '
        'default will be standard output')
    cmdparser.add_argument(
        '--format',
        help='figure format(png|pdf|ps|eps|svg)')
    cmdparser.add_argument(
        '-t',
        '--time',
        help='print returns-times or returns-samples '
        'default will be returns-samples')
    args = cmdparser.parse_args(argv)
    format = args.format
    if args.output:
        outputfile = open(args.output, 'wb')
        if not format:
            # pdb.set_trace()
            format = os.path.splitext(args.output)[1]
            if not format:
                format = 'png'
    else:
        outputfile = sys.stdout

    input_dir = "../logs/log_svrg_tr_swim/"
    #plot_average_return_sample(args.key, input_dir + args.input + "/progress.csv" , outputfile)
    inputfiles = [input_dir + inputfile + "/progress.csv" for inputfile in
                  args.inputs]
    if args.time == 'r':
        plot_average_return(args.key, inputfiles[0], outputfile)
    elif args.time == 't':
        plot_multiple_average_return_time(args.key, inputfiles, outputfile)
    else:
        plot_multiple_average_return_sample(args.key, inputfiles, outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
