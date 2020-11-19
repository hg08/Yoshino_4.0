#======
#Module
#======
from scipy.stats import norm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
from random import randrange
import random
import copy

#=========
# Functions
#=========
def generate_rand_normal_numbers(N=4):
    """生成N个满足正态分布的随机数"""
    mu = 0
    sigma = 1 
    samp = np.random.normal(loc=mu,scale=sigma,size=N)
    return samp
def init_J(L,N):
    """This function normalize the generated array J[i,j,:] to sqrt(tmp), and
     then make sure the constraint J[i,j,:]*J[i,j,:]=N are satisfied exactly."""
    J = np.zeros((L,N,N)) # axis=1 labels the backward-nodes (标记后一层结点), while axis=2 labels the forward-nodes (标记前一层结点)
    # Set the first layer J_{0,x}^{y} = 0 (x, y are any index.)
    for i in range(1,L):
        for j in range(N):         
            #First, generate N random numbers
            J[i,j,:] = generate_rand_normal_numbers(N)
            # Add the constraint np.sum(J[i,j,:] * J[i,j,:]) = N.
            tmp = np.sum(J[i,j,:] * J[i,j,:])
            J[i,j,:] = (J[i,j,:]/np.sqrt(tmp)) * np.sqrt(N) 
    return J 
#def init_S(L,M,N):
#    S = np.ones((L,M,N))
#    for i in range(L):
#        for j in range(M):
#            S[i,j,:] = generate_coord(N)
#    print("[For testing the validation of S] Ratio of positive S and negative S: {:7.6f}".format(S[S<0].size / S[S>0].size))
#    return S
def init_S(M,L,N):
    S = np.ones((M,L,N))
    for mu in range(M):
        for l in range(L):
            S[mu,l,:] = generate_coord(N)
    print("[For testing the validation of S] Ratio of positive S and negative S: {:7.6f}".format(S[S<0].size / S[S>0].size))
    return S
def generate_coord(N):
    """Randomly set the initial coordinates,whose components are 1 or -1."""
    v = np.ones(N)
    list_spin = [-1,1]
    for i in range(N):
        v[i] = np.random.choice(list_spin)
    return v
def generate_J(N):
    J = generate_rand_normal_numbers(N)
    return J
def soft_core_potential(h):
    '''Ref: Yoshino2019, eqn (32)
       This function is tested and it is correct.
    '''
    x2 = 1.0
    epsilon = 1
    return epsilon * (h**2) * np.heaviside(-h, x2) 
def calc_ener(r):
    '''Ref: Yoshino2019, eqn (31a)'''
    H = soft_core_potential(r).sum()
    return H

def ener_for_mu(r_mu):
    '''Ref: Yoshino2019, eqn (31a)
       energy for single sample (mu).'''
    H_mu = soft_core_potential(r_mu).sum()
    return H_mu
#======================================================================================================
#The following three functions are used for get the argument of locations of the initial configurations
#======================================================================================================
def list_only_dir(directory):
    """This function list the directorys only under a given direcotry."""
    import os
    list_dir = next(os.walk(directory))[1]
    full_li = []
    #directory = '../data'
    for item in list_dir:
        li = [directory,item]
        full_li.append("/".join(li))
    return full_li
def list_only_naked_dir(directory):
    """This function list the naked directory names only under a given direcotry."""
    import os
    list_dir = next(os.walk(directory))[1]
    for item in list_dir:
        li = [directory,item]
    return list_dir
def str2int(list_dir):
    res = []
    for item in range(len(list_dir)):
        res.append(int(list_dir[item]))
    return res
