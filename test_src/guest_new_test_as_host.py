#======
#Module
#======
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from random import choice
import scipy as sp
from scipy.stats import norm
from time import time
from utilities import *

class guest_network:
    def __init__(self,timestamp):
        #===========================================================================
        # This is host machine 1 in Yoshino_4.0. The intial configurations are from 
        # the super-host machine. The S and J are the coordinates of the machine.
        # In host machine, we have intialized the S and J in the following way:
        # S = init_S(M,L,N)
        # J = init_J(L,N)
        #===========================================================================
        """Since Yoshino_3.0, when update the energy, we do not calculate all the gaps, but only calculate the part affected by the flip of a SPIN (S)  or a shift of 
           a BOND (J). This will accelerate the calculation by hundreds of times. (2) Besides, we note that we do NOT need to define a functon: remain(), which records
           the new MC steps' S, J and H, even though one MC move is rejected."""
        # to obtain the basic integer parameters: M, L, N, tot_steps, from super-host
        para_list = np.load('../data/{:s}/para_list_basic.npy'.format(timestamp))
        beta_tmp = np.load('../data/{:s}/para_list_beta.npy'.format(timestamp))
        L = para_list[0]
        M = para_list[1]
        N = para_list[2]
        tot_steps = para_list[-1]
        # inverse temperature (beta)
        beta = beta_tmp[0]
        # store these parameters in the current machine
        self.L = L
        self.M = M 
        self.N = N 
        self.tot_steps = tot_steps
        self.beta = beta
        
        # Define new parameters; T (technically required)
        T = int(tot_steps+1)  # we keep the initial state in the first step 

        #Three arrays for storing MC trajectories of S, J and H
        self.J_traj = np.zeros((T, self.L, self.N, self.N)) 
        self.S_traj = np.zeros((T, self.M, self.L, self.N))
        self.H_traj = np.zeros(T)
        
        self.H = 0 # for storing energy when update
        self.new_H = 0 # for storing temperay energy when update

        # Energy difference caused by update of sample mu
        self.delta_H= 0 

        # For recording which kind of coordinate is activated
        self.S_active = False 
        self.J_active = False 
        #Intialize S and J by the array S and J. 
        #Note: Both S and J are the coordinates of a machine.
        self.S = np.load('../data/{:s}/seed_S_L{:d}_M{:d}_N{:d}_beta{:3.1f}.npy'.format(timestamp,L,M,N,beta))
        self.J = np.load('../data/{:s}/seed_J_L{:d}_N{:d}_beta{:3.1f}.npy'.format(timestamp,L,N,beta))
        self.new_S = copy.deepcopy(self.S) # for storing temperay array when update 
        self.new_J = copy.deepcopy(self.J) # for storing temperay array when update  
        self.r = self.gap_init() # the initial gap

        # index_for_* is used for defining the location to update in the DNN
        self.index_for_S = np.load('../data/{:s}/index_for_S_L{:d}_M{:d}_N{:d}.npy'.format(timestamp,L,M,N))
        self.index_for_J = np.load('../data/{:s}/index_for_J_L{:d}_N{:d}.npy'.format(timestamp,L,N))

        # series_for_x is used for shifting J
        self.series_for_x = np.load('../data/{:s}/series_for_x_L{:d}_N{:d}.npy'.format(timestamp,L,N))
        
        # series_for_decision_on_ is used for making a decision based on delta_E
        self.series_for_decision_on_S = np.load('../data/{:s}/series_for_decision_on_S_L{:d}_N{:d}.npy'.format(timestamp,L,N))
        self.series_for_decision_on_J = np.load('../data/{:s}/series_for_decision_on_J_L{:d}_N{:d}.npy'.format(timestamp,L,N))
        
        # Initialize the inner parameters: num_bonds, num_variables, num_nodes
        self.num_nodes = 0
        self.num_variables = 0
        self.num_bonds = 0
        self.num = 0

        self.count_MC_step = 0           

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
        # update self.new_S
        self.new_S = copy.deepcopy(self.S)
        self.new_S[mu][l][n] = -1 * self.S[mu][l][n]  
        # change the active-or-not state of S 
        self.S_active = True 
    def shift_bond(self,l,n2,n1,x):
        '''shift_bond() will shift the element of J with a given index to another value. We add l,n2,n1 as parameters, for parallel programming..'''
        self.new_J = copy.deepcopy(self.J)
        # scale denotes standard deviation; 
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J[l][n2][n1] = (self.J[l][n2][n1] + x * rat) / RESCALE_J
        # change the active-or-not state of J
        self.J_active = True 
    def accept_by_mu_l_n(self,mu,l,n):
        """This accept function is used if S is flipped."""
        self.S[mu,l,n] = self.new_S[mu,l,n]
        self.H = self.H + self.delta_H
    def accept_by_l_n2_n1(self,l,n2,n1):
        self.J[l,n2,n1] = self.new_J[l,n2,n1]
        self.H = self.H + self.delta_H
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

    def decision_by_mu_l_n(self,MC_index,mu,l,n,rand):
        self.delta_H = calc_ener(self.part_gap_after_flip(mu,l,n)) - calc_ener(self.part_gap_before_flip(mu,l,n))
        delta_e = round(self.delta_H,10)
        #delta_e = self.delta_H
        if delta_e < 0:
            self.accept_by_mu_l_n(mu,l,n) 
            print("Delta E:{:12.10f}".format(delta_e))
        else:
            if rand < np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n(mu,l,n)
                print("Delta E:{:12.10f}".format(delta_e))
            else:
                #pass # We do not need a "remain" function
                print("Delta E:{:12.10f}".format(delta_e))
    def decision_by_l_n2_n1(self,MC_index,l,n2,n1,rand):
        self.delta_H = calc_ener(self.part_gap_after_shift(l,n2)) - calc_ener(self.part_gap_before_shift(l,n2))
        delta_e = self.delta_H
        if delta_e < 0:
            # replace o.S by o.new_S:
            self.accept_by_l_n2_n1(l,n2,n1) 
            print("[J] Delta E:{:12.10f}".format(delta_e))
        else:
            if rand < np.exp(-delta_e * self.beta):
                self.accept_by_l_n2_n1(l,n2,n1)
                print("[J] Delta E:{:12.10f}".format(delta_e))
            else:
                print("[J] Delta E:{:12.10f}".format(delta_e))
                #pass # We do not need a "remain" function

if __name__=='__main__':
    #Parameters for rescaling J
    rat = 0.1 # r: Yoshino2019 Eq(35)
    RESCALE_J = np.sqrt(1+rat**2)
    MC_index = 0

    #=================================================================================
    # start_timestamp has two functions: 1st, calculate the time consumed by the code;
    # 2nd, used as the name of directory where the parameters will be located.
    #=================================================================================
    # Obtain the timestamp list
    start = time()
    start_time_int = int(time())
    data_dir = '../data'
    timestamp_list = list_only_naked_dir(data_dir)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-J', nargs='?', const=0, type=int, default=0, \
                        help="index of timestamp (integer).")
    # The time stamp 
    args = parser.parse_args()
    J = args.J

    timestamp = timestamp_list[J] # J is a index, but this index should given by job.sh

    # Initilize an instance of network.
    o = guest_network(timestamp)

    # define some parameters
    SQRT_N = np.sqrt(o.N)
    o.num_nodes = int(o.N*o.M*o.L)
    o.num_variables = int(o.N*o.M*(o.L-2))
    o.num_bonds = int(o.N*o.N*(o.L-1))
    num_variables = o.num_variables
    num_bonds = o.num_bonds
    num = o.num    

    o.S_traj[0,:,:,:] = o.S # Note that self.S_traj will independent of self.S from now on.
    o.J_traj[0,:,:,:] = o.J 
    o.H = calc_ener(o.r) # the energy
    o.H_traj[1] = o.H # H_traj[0] will be neglected
    tot_steps = o.tot_steps

    # MC 
    # For save J and S sequences
    name_S_seq = '../data/{}/seq_S_new_host_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.csv'.format(timestamp,o.L,o.M,o.N,o.beta,tot_steps)
    name_J_seq = '../data/{}/seq_J_new_host_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.csv'.format(timestamp,o.L,o.M,o.N,o.beta,tot_steps)
    file_o_S_seq = open(name_S_seq, 'w')
    file_o_J_seq = open(name_J_seq, 'w')
    # MC siulation starts
    for MC_index in range(1,tot_steps):
        print("MC step:{:d}".format(MC_index))
        #print("Updating S:")
        for update_index in range((MC_index-1)*num_variables, MC_index*num_variables):
            #mu,l,n = randrange(o.M), randrange(1,o.L-1), randrange(o.N)
            #o.flip_spin(mu,l,n)
            mu,l,n = o.index_for_S[update_index][0],o.index_for_S[update_index][1],o.index_for_S[update_index][2]
            o.flip_spin(mu,l,n)
            o.decision_by_mu_l_n(MC_index,mu,l,n,o.series_for_decision_on_S[update_index])
        #print("Updating J:")
        for update_index2 in range((MC_index-1)*num_bonds, MC_index*num_bonds):
            #l,n2,n1 = randrange(1,o.L),randrange(o.N),randrange(o.N)
            #o.shift_bond(l,n2,n1) 
            l,n2,n1,x = o.index_for_J[update_index2][0],o.index_for_J[update_index2][1],o.index_for_J[update_index2][2],o.series_for_x[update_index2]
            o.shift_bond(l,n2,n1,x) 
            o.decision_by_l_n2_n1(MC_index,l,n2,n1,o.series_for_decision_on_J[update_index2])
            print("Updating J:")
        o.count_MC_step += 1
        o.S_traj[MC_index] = o.S  
        o.J_traj[MC_index] = o.J
        o.H_traj[MC_index] = o.H
    # MC is done, we close files for recording the S and J sequences.
    file_o_S_seq.close()
    file_o_J_seq.close()
    
    np.save('../data/{:s}/S_new_host_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(timestamp,o.L,o.M,o.N,o.beta,tot_steps),o.S_traj)
    np.save('../data/{:s}/J_new_host_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(timestamp,o.L,o.N,o.beta,tot_steps),o.J_traj)
    np.save('../data/{:s}/ener_new_host_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(timestamp,o.L,o.M,o.N,o.beta,tot_steps),o.H_traj)
    
    # Finished
    print("All MC simulations done!")
    print('Time taken to run: {:5.1f} seconds.'.format(int(time())-start_time_int))
