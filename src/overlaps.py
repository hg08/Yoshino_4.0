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

def plot_overlap_J(overlap_J,L,N,beta,tot_steps):
    ex = 0 
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((0,1.0))
    x = -1 # Define a tempary integer
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(overlap_J[x+1][ex:],"navy",label="l=1")
    ax.plot(overlap_J[x+2][ex:],"blue",label="l=2")
    ax.plot(overlap_J[x+3][ex:],"royalblue",label="l=3")
    ax.plot(overlap_J[x+4][ex:],"khaki",label="l=4")
    ax.plot(overlap_J[x+5][ex:],"yellow",label='l=5')
    ax.plot(overlap_J[x+6][ex:],"gold",label='l=6')
    ax.plot(overlap_J[x+7][ex:],"plum",label='l=7')
    ax.plot(overlap_J[x+8][ex:],"purple",label="l=8")
    ax.plot(overlap_J[x+9][ex:],"m",label="l=9")
    ax.plot(overlap_J[x+10][ex:],"crimson",label='l=10')
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,l)$")
    ax.set_title("L={:d}; N={:d}; beta={:3.1f}".format(L,N,beta))
    plt.savefig("../imag/Overlap_J_L{:d}_N{:d}_beta{:3.1f}_step{:d}.eps".format(L,N,beta,tot_steps),format='eps')

def plot_overlap_S(overlap_S,L,M,N,beta,tot_steps):
    ex = 0
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((0,1.0))
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(overlap_S[1][ex:],"navy",label="l=1")
    ax.plot(overlap_S[2][ex:],"blue",label="l=2")
    ax.plot(overlap_S[3][ex:],"royalblue",label="l=3")
    ax.plot(overlap_S[4][ex:],"khaki",label="l=4")
    ax.plot(overlap_S[5][ex:],"yellow",label='l=5')
    ax.plot(overlap_S[6][ex:],"gold",label='l=6')
    ax.plot(overlap_S[7][ex:],"plum",label='l=7')
    ax.plot(overlap_S[8][ex:],"purple",label="l=8")
    ax.plot(overlap_S[9][ex:],"m",label="l=9")
    plt.legend(loc="upper right")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,l)$")
    ax.set_title("L={:d}; M={:d}; N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/Overlap_S_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.eps".format(L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Overlap_S_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.png".format(L,M,N,beta,tot_steps),format='png')

def overlap_J(J_traj,J0_traj):
    '''overlap for J_traj and J0_traj Q(t,l).'''
    N = J_traj.shape[-1]
    L = J_traj.shape[1]
    T = J_traj.shape[0]
    res = np.zeros((L,T))
    for l in range(L):
        for t in range(T):
            res[l][t] = np.sum(J_traj[t,l,:,:] * J0_traj[t,l,:,:])/(N**2)
    return res

def overlap_S(S_traj,S0_traj):
    '''overlap for S_traj and S0_traj q(t,l).'''
    L = S_traj.shape[1]
    M = S_traj.shape[2]
    N = S_traj.shape[-1]
    T = S_traj.shape[0]
    res = np.zeros((L,T))
    for l in range(1,L-1): # we do not need the spins in the input layer
        for t in range(T):
            res[l][t] = np.sum(S_traj[t,l,:,:] * S0_traj[t,l,:,:])/(N*M)
    return res
def overlap_JJ0(JJ0_traj):
    '''overlap for J_traj and J0_traj Q(t,l).'''
    N = JJ0_traj.shape[-1]
    L = JJ0_traj.shape[1]
    T = JJ0_traj.shape[0]
    res = np.zeros((L,T))
    for l in range(L):
        for t in range(T):
            res[l][t] = np.sum(JJ0_traj[t,l,:,:])/(N**2)
    return res

def overlap_SS0(SS0_traj):
    '''overlap for S_traj and S0_traj q(t,l).'''
    L = SS0_traj.shape[1]
    M = SS0_traj.shape[2]
    N = SS0_traj.shape[-1]
    T = SS0_traj.shape[0]
    res = np.zeros((L,T))
    for l in range(1,L-1): # we do not need the spins in the input layer
        for t in range(T):
            res[l][t] = np.sum(SS0_traj[t,l,:,:])/(N*M)
    return res

if __name__ == '__main__':
    nk = 1 # Before running, please set the number of samples by hand
    import argparse
    mpl.use('Agg')
    ext_index = 0 
    
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir)
    #-----------------------------------------------------------------------------------------------------------------
    #Since all the simulations,we use the same M,L,N,tot_steps. we can obtain these parameter from the fisrt direcotry
    #The following 5 lines do this thing. 
    #-----------------------------------------------------------------------------------------------------------------
    para_list = np.load('{}/{}/para_list_basic.npy'.format(data_dir,timestamp_list[0]))
    L = para_list[0]
    M = para_list[1]
    N = para_list[2]
    tot_steps = para_list[-1]
    
    high_dim_J = np.zeros((nk,L,tot_steps))
    high_dim_S = np.zeros((nk,L,tot_steps))
    ave_traj_JJ0 = np.zeros((tot_steps+1,L,N,N))
    ave_traj_SS0 = np.zeros((tot_steps+1,L+1,M,N))
    for i in range(nk):
        beta_tmp = np.load('{}/{}/para_list_beta.npy'.format(data_dir,timestamp_list[i]))
        beta = beta_tmp[0]
        J = np.load('{}/{}/J_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],L,N,beta,tot_steps))
        J0 = np.load('{}/{}/J_guest_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],L,N,beta,tot_steps))
        S = np.load('{}/{}/S_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],L,M,N,beta,tot_steps))
        S0 = np.load('{}/{}/S_guest_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data_dir,timestamp_list[i],L,M,N,beta,tot_steps))
        ave_traj_JJ0 = ave_traj_JJ0 + J * J0
        ave_traj_SS0 = ave_traj_SS0 + S * S0
        print(J[0,:,:,:])
        print(J0[0,:,:,:])
    ave_traj_JJ0 = ave_traj_JJ0/nk
    ave_traj_SS0 = ave_traj_SS0/nk
    #ave_traj_J0 = ave_traj_J0/nk
    #ave_traj_S0 = ave_traj_S0/nk
    #Remember: Do not use a function'name as a name of variable
    res_J = overlap_JJ0(ave_traj_JJ0) 
    res_S = overlap_SS0(ave_traj_SS0) 
    #=====================================================
    # Do the average over different initial configurations.
    # np.mean() can average over nk, the first axis, 
    # therefore, 'axis=0' is used.
    #=====================================================
    #plot_ave_overlap_J(np.mean(high_dim_J,axis=0),L,N,beta,tot_steps)
    #plot_ave_overlap_S(np.mean(high_dim_S,axis=0),M,L,N,beta,tot_steps) 
    plot_overlap_J(res_J,L,N,beta,tot_steps)
    plot_overlap_S(res_S,L,M,N,beta,tot_steps) 
