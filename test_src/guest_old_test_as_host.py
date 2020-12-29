#======
#Module
#======
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
import scipy as sp
from scipy.stats import norm
from time import time
from utilities_old import *

class guest_network:
    def __init__(self,timestamp):
        #===========================================================================
        # S and J are the coordinates of the machine.
        # This is a host machine in Yoshino_4.0. The intial configurations are from 
        # the generator machine. 
        # In host machine, we have intialized the S and J in the following way:
        # S = init_S(M,L,N)
        # J = init_J(L,N)
        #===========================================================================
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
        self.S_traj = np.zeros((T, self.M, self.L, self.N))
        self.J_traj = np.zeros((T, self.L, self.N, self.N))
        self.H_traj = np.zeros(T)

        self.H = 0 # the total energy
        self.new_H = 0 # for storing temperay energy when update

        # For recording which kind of coordinate is activated
        self.S_active = False 
        self.J_active = False 
        #Intialize S and J by the array S and J used in the teacher machine.
        #Note: S and J are the coordinates of a machine.
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

        # Initialize the simulation steps
        self.count_MC_step = 0           

    def gap_init(self):
        '''Ref: Yoshino2019, eqn (31b)'''
        r = np.ones((self.M,self.L,self.N))
        for mu in range(self.M):
            for l in range(1,self.L): # l = 0,...,L-1
                for n in range(self.N):
                    r[mu,l,n] = (np.sum(self.J[l,n,:] * self.S[mu,l-1,:])/np.sqrt(self.N)) * self.S[mu,l,n] 
        return r    
    def gap(self):
        '''Ref: Yoshino2019, eqn (31b)'''
        #r = np.ones((self.M,self.L,self.N))
        for mu in range(self.M):
            for l in range(1,self.L):
                for n in range(self.N):
                    self.r[mu,l,n] = (np.sum(self.new_J[l,n,:] * self.new_S[mu,l-1,:])/np.sqrt(self.N)) * self.new_S[mu,l,n] 
    def ener_after_flip(self):
        temp_r = np.ones((self.M,self.L,self.N))
        for mu in range(self.M):
            for l in range(1,self.L):
                for n in range(self.N):
                    temp_r[mu,l,n] = (np.sum(self.J[l,n,:] * self.new_S[mu,l-1,:])/np.sqrt(self.N)) * self.new_S[mu,l,n]
        ener = calc_ener(temp_r) 
        return ener 
    def ener_after_shift(self):
        temp_r = np.ones((self.M,self.L,self.N))
        for mu in range(self.M):
            for l in range(1,self.L):
                for n in range(self.N):
                    temp_r[mu,l,n] = (np.sum(self.new_J[l,n,:] * self.S[mu,l-1,:])/np.sqrt(self.N)) * self.S[mu,l,n]
        ener = calc_ener(temp_r) 
        return ener 
    def flip_spin(self,mu,l,n):
        '''flip_spin() will generate a new array new_S. Note: all the spins can be update except the input/output.'''
        # update self.new_S
        self.new_S = copy.deepcopy(self.S)
        self.new_S[mu][l][n] = -1 * self.new_S[mu][l][n]  
     
    def shift_bond(self,l,n2,n1,x):
        '''shift_bond() will generate a new array new_J.'''
        self.new_J = copy.deepcopy(self.J)
        # scale denotes standard deviation; 
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J[l][n2][n1] = (self.J[l][n2][n1] + x * rat) / RESCALE_J
    def accept(self):
        if self.S_active == True:
            self.S = copy.deepcopy(self.new_S)
        else:
            self.J = copy.deepcopy(self.new_J)
        self.H = copy.deepcopy(self.new_H) 
        #print("ENERGY:{}".format(self.H))

    def decision(self,MC_index,rand):
        """
        1. np.random.random(1) generate a random float number between 0 and 1.
        2. accept probability=min(1, exp(-\beta \Delta_E))
        3. k_B = 1.
        """
        if self.S_active == True:
            #self.gap() # update self.r
            self.new_H = o.ener_after_flip()
        else:
            #self.gap() # update self.r
            self.new_H = o.ener_after_shift()
        delta_e = round(self.new_H - self.H, 10)
        #delta_e = self.new_H - self.H
        if delta_e < 0:
            # replace o.S by o.new_S: use copy.deepcopy() 
            self.accept() 
            #print("ACCEPTED.")
            print("Delta E:{:12.10f}".format(delta_e))
        else:
            if rand < np.exp(-delta_e * self.beta):
                self.accept()       
                print("Delta E:{:12.10f}".format(delta_e))
                #print("ACCEPTED.")
            else:
                print("Delta E:{:12.10f}".format(delta_e))
# =====
if __name__=='__main__':
    #parameters
    rat = 0.1 # r: Yoshino2019 Eq(35)
    RESCALE_J = np.sqrt(1+rat**2)
    # Initialization
    MC_index = 0

    # Obtain the timestamp list
    # Startint time
    start = time()
    data_dir = '../data'
    timestamp_list = list_only_naked_dir(data_dir)
    #timestamp_list = str2int(data_list)

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
    o.num_variables = int(o.N*o.M*(o.L-2))
    o.num_bonds = int(o.N*o.N*(o.L-1))

    o.S_traj[0,:,:,:] = o.S # Note that self.S_traj will independent of self.S from now on.
    o.J_traj[0,:,:,:] = o.J 
    o.H = calc_ener(o.r) # the energy
    o.H_traj[1] = o.H # H_traj[0] will be neglected
    #print("ENergy: first step:")
    #print(o.H)
    #print(o.H_traj[1])
    tot_steps = o.tot_steps

    # MC 
    # For save J and S sequences
    name_S_seq = '../data/{}/seq_S_old_host_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.csv'.format(timestamp,o.L,o.M,o.N,o.beta,tot_steps)
    name_J_seq = '../data/{}/seq_J_old_host_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.csv'.format(timestamp,o.L,o.M,o.N,o.beta,tot_steps)
    file_o_S_seq = open(name_S_seq, 'w')
    file_o_J_seq = open(name_J_seq, 'w')

    for MC_index in range(1,tot_steps):
        print("MC step:{:d}".format(MC_index))
        #print("Updating S:")
        o.S_active = True 
        o.J_active = False 
        for update_index in range((MC_index-1)*o.num_variables, MC_index*o.num_variables):
            # Flip one spin and make a decision: there are M*(L-1)*N times
            mu,l,n = o.index_for_S[update_index][0],o.index_for_S[update_index][1],o.index_for_S[update_index][2]
            o.flip_spin(mu,l,n)
            o.decision(MC_index,o.series_for_decision_on_S[update_index])

        o.S_active = False 
        o.J_active = True 
        #print("Updating J:")
        for update_index in range((MC_index-1)*o.num_bonds, MC_index*o.num_bonds):
            # shift one bond (interaction) and make a decision: there are L*N*N times
            l,n2,n1,x = o.index_for_J[update_index][0],o.index_for_J[update_index][1],o.index_for_J[update_index][2],o.series_for_x[update_index]
            o.shift_bond(l,n2,n1,x)
            o.decision(MC_index,o.series_for_decision_on_J[update_index])
            print("Updating J:")
            
        o.count_MC_step += 1
        o.S_traj[MC_index] = o.S 
        o.J_traj[MC_index] = o.J
        o.H_traj[MC_index] = o.H

    # MC is done, we close files for recording the S and J sequences.
    file_o_S_seq.close()
    file_o_J_seq.close()
    
    np.save('../data/{:s}/S_old_host_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(timestamp,o.L,o.M,o.N,o.beta,tot_steps),o.S_traj)
    np.save('../data/{:s}/J_old_host_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(timestamp,o.L,o.N,o.beta,tot_steps),o.J_traj)
    np.save('../data/{:s}/ener_old_host_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(timestamp,o.L,o.M,o.N,o.beta,tot_steps),o.H_traj)
    
    # Finished
    print("MC simulations are done.")
    print(f'Time taken to run: {time() - start} seconds')
