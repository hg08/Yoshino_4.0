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
def get_N_HexCol(N=5):
    """Define N elegant colors and return a list of the N colors. Each element of the list is represented as a string.
       and it can be used as an argument of the kwarg color in plt.plot(), or plt.hist()."""
    import colorsys
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out
def plot_overlap_J(overlap_J,L,N,beta,tot_steps):
    ex = 0 
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((0,1.0))
    x = 0 # Define a tempary integer
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(overlap_J[x+1][ex:],color_list[0],label="l=1")
    ax.plot(overlap_J[x+2][ex:],color_list[1],label="l=2")
    ax.plot(overlap_J[x+3][ex:],color_list[2],label="l=3")
    ax.plot(overlap_J[x+4][ex:],color_list[3],label="l=4")
    ax.plot(overlap_J[x+5][ex:],color_list[4],label='l=5')
    ax.plot(overlap_J[x+6][ex:],color_list[5],label='l=6')
    ax.plot(overlap_J[x+7][ex:],color_list[6],label='l=7')
    ax.plot(overlap_J[x+8][ex:],color_list[7],label="l=8")
    ax.plot(overlap_J[x+9][ex:],color_list[8],label="l=9")
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,l)$")
    ax.set_title("L={:d}; N={:d}; beta={:3.1f}".format(L,N,beta))
    plt.savefig("../imag/Overlap_J_L{:d}_N{:d}_beta{:3.1f}_step{:d}.eps".format(L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Overlap_J_L{:d}_N{:d}_beta{:3.1f}_step{:d}.png".format(L,N,beta,tot_steps),format='png')

def plot_overlap_S(overlap_S,L,M,N,beta,tot_steps):
    ex = 0
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((0,1.0))
    x = 0 
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(overlap_S[x+1][ex:],color_list[0],label="l=1")
    ax.plot(overlap_S[x+2][ex:],color_list[1],label="l=2")
    ax.plot(overlap_S[x+3][ex:],color_list[2],label="l=3")
    ax.plot(overlap_S[x+4][ex:],color_list[3],label="l=4")
    ax.plot(overlap_S[x+5][ex:],color_list[4],label='l=5')
    ax.plot(overlap_S[x+6][ex:],color_list[5],label='l=6')
    ax.plot(overlap_S[x+7][ex:],color_list[6],label='l=7')
    ax.plot(overlap_S[x+8][ex:],color_list[7],label="l=8")
    ax.plot(overlap_S[x+9][ex:],color_list[8],label="l=9")
    plt.legend(loc="upper right")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,l)$")
    ax.set_title("L={:d}; M={:d}; N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/Overlap_S_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.eps".format(L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Overlap_S_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.png".format(L,M,N,beta,tot_steps),format='png')

def overlap_J(J_traj,J0_traj):
    '''overlap for J_traj and J0_traj Q(t,l).'''
    T = J_traj.shape[0]
    L = J_traj.shape[1]
    N = J_traj.shape[-1]
    res = np.zeros((L,T))
    for l in range(L):
        for t in range(T):
            res[l][t] = np.sum(J_traj[t,l,:,:] * J0_traj[t,l,:,:])/(N**2)
    return res

def overlap_S(S_traj,S0_traj):
    '''overlap for S_traj and S0_traj q(t,l).'''
    T = S_traj.shape[0]
    M = S_traj.shape[1]
    L = S_traj.shape[2]
    N = S_traj.shape[-1]
    res = np.zeros((L,T))
    for l in range(1,L-1): # we do not need the spins in the input layer
        for t in range(T):
            res[l][t] = np.sum(S_traj[t,:,l,:] * S0_traj[t,:,l,:])/(N*M)
    return res
def overlap_JJ0(JJ0_traj):
    '''overlap for J_traj and J0_traj Q(t,l).'''
    T = JJ0_traj.shape[0]
    L = JJ0_traj.shape[1]
    N = JJ0_traj.shape[-1]
    res = np.zeros((L,T))
    for l in range(1,L):
        for t in range(T):
            res[l][t] = np.sum(JJ0_traj[t,l,:,:])/(N**2)
    return res

def overlap_SS0(SS0_traj):
    '''overlap for S_traj and S0_traj q(t,l).'''
    T = SS0_traj.shape[0]
    M = SS0_traj.shape[1]
    L = SS0_traj.shape[2]
    N = SS0_traj.shape[-1]
    res = np.zeros((L,T))
    for l in range(1,L-1): # we do not need the spins in the input layer
        for t in range(T):
            res[l][t] = np.sum(SS0_traj[t,:,l,:])/(N*M)
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
    ave_traj_SS0 = np.zeros((tot_steps+1,M,L,N))
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
