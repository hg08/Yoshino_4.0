#======
#Module
#======
import os
from time import time
import multiprocessing as mp
from scipy.stats import norm
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from random import randrange
from random import choice
import copy
from utilities import *

class host_network:
    def __init__(self,L,M,N,tot_steps,beta,timestamp):
        """Since Yoshino_3.0, when update the energy, we do not calculate energy for all the gaps, but only calculate these part affected by the flipping of a SPIN (S)  or shifting of 
           a BOND (J). This will accelerate the calculation by hundreds of times. (2) Besides, we note that we do NOT need to define a functon: remain(), which records
           the new MC steps' S, J and H, even though one MC move is rejected."""
        # the parameters used in the host machine
        self.L = int(L) # number of layers
        self.M = int(M) # number of samples
        self.N = int(N) # the number of neurons at each layer is a constant.
        self.tot_steps = int(tot_steps) # number of total MC simulation steps
        self.timestamp = timestamp  
        self.beta = beta # inverse temperature
        
        # Define new parameters: T (technically required)
        T = self.tot_steps+1  # we keep the initial state in the first step 

        # The arrays for storing MC trajectories of S, J and H
        self.J_traj = np.zeros((T, self.L, self.N, self.N)) 
        self.S_traj = np.zeros((T, self.M, self.L, self.N))
        self.H_traj = np.zeros(T)
        
        self.H = 0 # for storing energy when update
        self.new_H = 0 # for storing temperay energy when update

        # Energy difference caused by update of variables J or S 
        self.delta_H= 0 

        # For recording which kind of coordinate is activated
        self.S_active = False 
        self.J_active = False 
        #Intialize S and J by the array S and J. 
        #Note: Both S and J are the coordinates of a machine.
        self.S = init_S(self.M, self.L, self.N) # L layers
        self.J = init_J(self.L, self.N) # L layers 
        self.r = self.gap_init() # the initial gap is returned from a function.

        self.new_S = copy.deepcopy(self.S) # for storing temperay array when update 
        self.new_J = copy.deepcopy(self.J) # for storing temperay array when update  
        self.new_r = copy.deepcopy(self.r)

        self.count_MC_step = 0           

        # For recording which layer is updating
        self.updating_layer_index = None 
        # For recording which node is updating for S
        self.updating_node_index = None 
        # For recording which node in forward layer is updating for J
        self.updating_node_index_n1 = None 
        # For recording which node in backward layer is updating for J
        self.updating_node_index_n2 = None 
        # For recording which sample is updating
        self.updating_sample_index = None

    def gap_init(self):
        '''Ref: Yoshino2019, eqn (31b)'''
        r = np.zeros((self.M,self.L,self.N))
        for mu in range(self.M):
            for l in range(1,self.L): # l = 0,...,L-1
                for n2 in range(self.N):
                    r[mu,l,n2] = (np.sum(self.J[l,n2,:] * self.S[mu,l-1,:])/np.sqrt(self.N)) * self.S[mu,l,n2] 
        return r    
    def flip_spin(self,mu,l,n):
        '''flip_spin() will flip S at a given index tuple (l,mu,n). We add l,mu,n as parameters, for parallel programming. Note: any spin can be update except the input/output.'''
        # record this step
        # update self.new_S
        self.new_S = copy.deepcopy(self.S)
        self.new_S[mu][l][n] = -1 * self.S[mu][l][n]  
        # change the active-or-not state of S 
        self.S_active = True 
    def shift_bond(self,l,n2,n1):
        '''shift_bond() will shift the element of J with a given index to another value. We add l,n2,n1 as parameters, for parallel programming..'''
        self.new_J = copy.deepcopy(self.J)
        x = np.random.normal(loc=0,scale=1.0,size=1) 
        # scale denotes standard deviation; 
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J[l][n2][n1] = (self.J[l][n2][n1] + x * rat) / RESCALE_J
        # change the active-or-not state of J
        self.J_active = True 
    def accept_by_mu_l_n(self,mu,l,n):
        """This accept function is used if S is flipped."""
        self.S[mu,l,n] = self.new_S[mu,l,n]
        self.H = self.H + self.delta_H
        print("ENERGY: {}".format(self.H))
    def accept_by_l_n2_n1(self,l,n2,n1):
        """This accept function is used if J is shifted."""
        self.J[l,n2,n1] = self.new_J[l,n2,n1]
        self.H = self.H + self.delta_H
        print("ENERGY: {}".format(self.H))
    def part_gap_before_flip(self,mu,l,n):
        '''Ref: Yoshino2019, eqn (31b)
           When S is fliped, only one machine changes its coordinates and it will affect the gap of the node before it and the gaps of the N nodes
           behind it. Therefore, N+1 gaps contributes to the Delta_H_eff. l = 0,1, ..., L-1. 
           We define a small array, part_gap, which has N+1 elements. Each elements of part_gap is a r^mu_node. Use part_gap, we can calculate the 
           Energy change coused by the flip of S^mu_node,n. 
        '''
        part_gap = np.zeros(self.N + 1)
        part_gap[0] = (np.sum( self.J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n] 
        for n2 in range(self.N):
            part_gap[1+n2] = (np.sum(self.J[l+1,n2,:] * self.S[mu,l,:])/SQRT_N) * self.S[mu,l+1,n2] 
        return part_gap  # Only the N+1 elements affect the Delta_H_eff. 
    def part_gap_after_flip(self,mu,l,n):
        part_gap = np.zeros(self.N + 1)
        part_gap[0] = (np.sum( self.J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.new_S[mu,l,n] 
        for n2 in range( self.N):
            part_gap[1+n2] = (np.sum(self.J[l+1,n2,:] * self.new_S[mu,l,:])/SQRT_N) * self.S[mu,l+1,n2] 
        return part_gap  # Only the N+1 elements affect the Delta_H_eff. 
    def part_gap_before_shift(self,l,n): 
        part_gap = np.zeros(self.M)
        for mu in range(self.M):
            part_gap[mu] = (np.sum(self.J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n] 
        return part_gap  # Only the M elements affect the Delta_H_eff. 
    def part_gap_after_shift(self,l,n): 
        part_gap = np.zeros(self.M)
        for mu in range(self.M):
            part_gap[mu] = (np.sum(self.new_J[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n] 
        return part_gap # Only the M elements affect the Delta_H_eff. 
    def decision_by_mu_l_n(self,MC_index,mu,l,n):
        self.delta_H = calc_ener(self.part_gap_after_flip(mu,l,n)) - calc_ener(self.part_gap_before_flip(mu,l,n))
        delta_e = self.delta_H
        print("[S] delta_E:{}".format(delta_e)) 
        if delta_e < 0:
            self.accept_by_mu_l_n(mu,l,n) 
            print("ACC.")       
        else:
            if np.random.random(1) < np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n(mu,l,n)
                print("ACC.")       
            else:
                pass
    def decision_by_l_n2_n1(self,MC_index,l,n2,n1):
        self.delta_H = calc_ener(self.part_gap_after_shift(l,n2)) - calc_ener(self.part_gap_before_shift(l,n2))
        delta_e = self.delta_H
        print("[J] delta_E:{}".format(delta_e)) 
        if delta_e < 0:
            self.accept_by_l_n2_n1(l,n2,n1) 
            print("ACCEPT.")       
        else:
            if np.random.random(1) < np.exp(-delta_e * self.beta):
                self.accept_by_l_n2_n1(l,n2,n1)
                print("ACCEPT.")       
            else:
                pass
    def update_spin(self,ind): 
        self.flip_spin(ind[0],ind[1],ind[2])
        self.decision_by_mu_l_n(MC_index,ind[0],ind[1],ind[2])

if __name__=='__main__':
    #Parameters for rescaling J
    rat = 0.1 # r: Yoshino2019 Eq(35)
    RESCALE_J = np.sqrt(1+rat**2)
    multiple_update = True 
    num_cpu = os.cpu_count()
    MC_index = 0

    #=================================================================================
    # start_timestamp has two functions: 1st, calculate the time sonsumed by the code;
    # 2nd, used as the name of directory where the parameters will be located.
    #=================================================================================
    start = time()
    start_timestamp = int(time())
    start_time_int = int(time())
    print("starting time:{}".format(start_time_int))
    # Read the arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', nargs='?', const=8, type=int, default=8, \
                        help="the number of samples.")
    parser.add_argument('-L', nargs='?', const=5, type=int, default=5, \
                        help="the number of hidden layers.(Condition: L > 1)") 
    parser.add_argument('-N', nargs='?', const=6, type=int, default=6, \
                        help="the number of nodes per layer.") 
    parser.add_argument('-S', nargs='?', const=10, type=int, default=10, \
                        help="the number of total steps.")
    parser.add_argument('-B', nargs='?', const=66.7, type=float, default=66.7, \
                        help="the inverse temperautre.") 
    args = parser.parse_args()
    M,N,L,beta,tot_steps = args.M, args.N, args.L,args.B, args.S

    # Preparing parameters for using in student machine
    parameter_list_basic = np.array([L,M,N,tot_steps])
    parameter_list_beta = np.array([beta])

    # Initilize an instance of network.
    o = host_network(L,M,N,tot_steps,beta,start_timestamp)
    # define some parameters
    SQRT_N = np.sqrt(o.N)
    num_nodes = int(o.N*o.M*o.L)
    num_variables = int(o.N*o.M*(o.L-2))
    num_bonds = int(o.N*o.N*(o.L-1))
    num = num_variables+num_bonds

    #=====================================
    #Important step
    #Save the seed for the guest machine
    #=====================================
    # make a new directory named start_timestamp
    start_timestamp = str(start_timestamp)
    list_dir = ['../data/',start_timestamp]
    data = "../data"
    name_dir = "".join(list_dir)
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)
    # save the arrays into the created directory
    np.save('{}/{:s}/para_list_basic.npy'.format(data,start_timestamp), parameter_list_basic)
    np.save('{}/{:s}/para_list_beta.npy'.format(data,start_timestamp), parameter_list_beta)
    np.save('{}/{:s}/seed_S_L{:d}_M{:d}_N{:d}_beta{:3.1f}.npy'.format(data,start_timestamp,L,M,N,beta),o.S)
    np.save('{}/{:s}/seed_J_L{:d}_N{:d}_beta{:3.1f}.npy'.format(data,start_timestamp,L,N,beta),o.J)
    o.S_traj[0,:,:,:] = o.S # Note that self.S_traj will independent of self.S from now on.
    o.J_traj[0,:,:,:] = o.J 
    o.H = calc_ener(o.r) # the energy
    o.H_traj[0] = o.H # H_traj[0] is the initial value of H

    # MC 
    # For save J and S sequences
    name_S_seq = '{}/{}/seq_S_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.csv'.format(data,start_timestamp,L,M,N,beta,tot_steps)
    name_J_seq = '{}/{}/seq_J_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.csv'.format(data,start_timestamp,L,M,N,beta,tot_steps)
    file_o_S_seq = open(name_S_seq, 'w')
    file_o_J_seq = open(name_J_seq, 'w')
    # MC siulation starts
    if multiple_update:
        for MC_index in range(1,tot_steps):
            print("MC step:{:d}".format(MC_index))
            print("Updating S:")
            ## If use parallel programing:
            #for update_index in range(num_variables):
            #    tmp_li = []
            #    for l in range(choice([0,1]),o.L,2):
            #        mu,n = randrange(o.M), randrange(o.N)
            #        tmp_li.append((l,mu,n))
            #    print("The active S index: {}".format(tmp_li))
            #    with mp.Pool(num_cpu) as p:
            #        p.map(o.update_spin, tmp_li)
            for update_index in range(num_variables):
                mu,l,n = randrange(o.M), randrange(1,o.L-1), randrange(o.N)
                o.flip_spin(mu,l,n)
                o.decision_by_mu_l_n(MC_index,mu,l,n)
            print("Updating J:")
            for update_index in range(num_bonds):
                l,n2,n1 = randrange(1,o.L),randrange(o.N),randrange(o.N)
                o.shift_bond(l,n2,n1) 
                o.decision_by_l_n2_n1(MC_index,l,n2,n1)
            o.count_MC_step += 1
            o.S_traj[MC_index] = o.S  #
            o.J_traj[MC_index] = o.J
            o.H_traj[MC_index] = o.H
    else:
        pass
    # MC is done, we can close some files for recording the S and J sequences.
    file_o_S_seq.close()
    file_o_J_seq.close()

    np.save('{}/{}/S_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data,start_timestamp,L,M,N,beta,tot_steps),o.S_traj)
    np.save('{}/{}/J_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data,start_timestamp,L,N,beta,tot_steps),o.J_traj)
    np.save('{}/{}/ener_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data,start_timestamp,L,M,N,beta,tot_steps),o.H_traj)
    
    # Finished
    print("All MC simulations done!")
    print('Time taken to run: {:5.1f} seconds.'.format(int(time())-start_time_int))
