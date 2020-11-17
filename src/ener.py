#=======
# Module
#=======
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from utilities import *

#==========
# Functions
#==========
def autocorr(x):
    """Correlation Function."""
    x = np.array(x)
    length = np.size(x)
    c = np.ones(length)
    for i in range(1,length):
        c[i]=np.sum(x[:-i]*x[i:])/np.sum(x[:-i]*x[:-i])
    return c
def line2intlist(line):
    line_split=line.strip().split(' ')
    res_list = []
    for x in line_split:
        res_list.append(int(x))
    return res_list
def line2floatlist(line):
    line_split=line.strip().split(' ')
    res_list = []
    for x in line_split:
        res_list.append(float(x))
    return res_list

def plot_ener_dynamics(L,M,N,beta,timestamp):
    ex = 1 
    data="../data1"
    ener = np.load('{}/{}/ener_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data,timestamp,L,M,N,beta,tot_steps))
    ener2 = np.load('{}/{}/ener_guest_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data,timestamp,L,M,N,beta,tot_steps))
    fig = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(ener[1:],"r-",label="a")
    ax.plot(ener2[1:],"b-",label='b')
    plt.legend(loc="upper right")
    plt.xlabel("$t$")
    plt.ylabel(r"$E(t)$")
    ax.set_title("L={:d}; M={:d}; N={:d}; beta={:3.1f}".format(L,M,N,beta),fontname="Arial", fontsize=12)
    plt.savefig("../imag/ener_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}_{}.eps".format(L,M,N,beta,tot_steps,timestamp),format='eps')

if __name__ == '__main__':
    import argparse
    mpl.use('Agg')
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir)
    nk = 1 
    for i in range(nk):
        para_list = np.load('{}/{}/para_list_basic.npy'.format(data_dir,timestamp_list[i]))
        beta_tmp = np.load('{}/{}/para_list_beta.npy'.format(data_dir,timestamp_list[i]))
        L = para_list[0]
        M = para_list[1]
        N = para_list[2]
        tot_steps = para_list[-1]
        beta = beta_tmp[0]
        plot_ener_dynamics(L,M,N,beta,timestamp_list[i])
