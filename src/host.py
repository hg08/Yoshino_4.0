#======
#Module
#======
import os
import multiprocessing as mp
from scipy.stats import norm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
from random import randrange
from random import choice
import copy
from utilities import *
from time import time

#=========
#The Class
#=========
class host_network:
    def __init__(self,N,M,L,tot_steps,beta,timestamp):
        """In this new class, in Yoshino_3.0, when we update the energy, we do not calculate all the gaps, but only calculate these affected by the flip of a SPIN (S)  or a shift of 
           a BOND (J). This will accelerate the calculation by hundreds of times. (2) Besides, we also have to note that we do NOT need to define a functon: remain(), which records
           the new MC steps' S, J and H, even though one MC move is rejected."""
        # store the parameters used in the host machine
        self.M = int(M) # number of samples
        self.L = int(L) # number of layers
        self.N = int(N) # the number of neurons at each layer.
        self.tot_steps = int(tot_steps) 
        self.timestamp = timestamp  
        self.beta = beta # inverse temperature
        
        # Define new parameters: T (technically required)
        self.T = int(tot_steps+1)  # we keep the initial state in the first step 

        # The arrays for storing MC trajectories of S, J; the Accept/Reject probability of S and J; H
        self.J_traj = np.zeros((self.T, self.L, self.N, self.N)) 
        self.S_traj = np.zeros((self.T, self.L+1, self.M, self.N))
        self.H_traj = np.zeros(self.T)
        
        self.H = 0 # for storing energy when update
        self.new_H = 0 # for storing temperay energy when update

        # Energy difference caused by update of sample mu
        self.delta_H= 0 

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

        # For recording which kind of coordinate is activated
        self.S_active = False 
        self.J_active = False 
        #Intialize S and J by the array S and J. 
        #Note: Both S and J are the coordinates of a machine.
        self.S = init_S(self.L+1, self.M, self.N) # L+1 layers
        self.J = init_J(self.L, self.N) # L layers 
        self.new_S = copy.deepcopy(self.S) # for storing temperay array when update 
        self.new_J = copy.deepcopy(self.J) # for storing temperay array when update  
        self.r = self.gap_init() # the initial gap is returned from a function.

        #self.new_r = self.r
        self.new_r = copy.deepcopy(self.r)

        # Initialize the simulation steps
        self.count_S_update = 0           
        self.count_J_update = 0           
        self.count_MC_step = 0           

    def gap_init(self):
        '''Ref: Yoshino2019, eqn (31b)'''
        r = np.ones((self.M,self.L,self.N))
        for mu in range(self.M):
            for l in range(self.L): # l = 0,...,L-1
                for n in range(self.N):
                    r[mu,l,n] = (np.sum(self.J[l,n,:] * self.S[l,mu,:])/np.sqrt(self.N)) * self.S[l+1,mu,n] 
        return r    
    def flip_S(self):
        '''flip_S() will generate a new array new_S. Note: all the spins can be update except the input/output.'''
        # record this step
        self.new_S = copy.deepcopy(self.S)
        #self.new_S = self.S 
        i,j,k = randrange(1,self.L), randrange(self.M), randrange(self.N)
        # update self.new_S
        #self.new_S[1+i][j][k] = -1 * self.new_S[1+i][j][k]  
        self.new_S[i][j][k] = -1 * self.S[i][j][k]  
        # record this step
        self.count_S_update += 1          
        # record the label of the updating sample mu =j (S)
        self.updating_sample_index = j 
        # record the label of the updating layer (S)
        self.updating_layer_index = i 
        # record the label of the updating node (S)
        self.updating_node_index = k 
        # change the active-or-not state of S 
        self.S_active = True 
    def shift_bond(self):
        '''shift_bond() will generate a new array new_J.'''
        self.new_J = copy.deepcopy(self.J)
        i,n2,n1 = randrange(self.L),randrange(self.N),randrange(self.N)
        x = np.random.normal(loc=0,scale=1.0,size=1) 
        # scale denotes standard deviation; 
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J[i][n2][n1] = (self.J[i][n2][n1] + x * rat) / RESCALE_J
        # record this step
        self.count_J_update += 1
        # record the label of the updating layer (J) 
        self.updating_layer_index = i 
        # record the label of the updating node 1 of a link (J)
        self.updating_node_index_n2 = n2 
        # record the label of the updating node 2 of a link (J)
        self.updating_node_index_n1 = n1
        # change the active-or-not state of J
        self.J_active = True 
    def flip_multiple_S(self,rand):
        '''flip_multiple_S() will generate a new array new_S. Note: all the spins can be update except the input/output.'''
        # record this step
        active_S_index = []
        self.new_S = copy.deepcopy(self.S)
        #self.new_S = self.S 
        for i in range(1+rand,self.L,2):
            j,k = randrange(self.M), randrange(self.N)
            #update self.new_S
            #self.new_S[1+i][j][k] = -1 * self.new_S[1+i][j][k]  
            self.new_S[i][j][k] = -1 * self.S[i][j][k] 
            active_S_index.append((i,j,k)) 
            # record this step
            self.count_S_update += 1          
        # change the active-or-not state of S 
        self.S_active = True
        return active_S_index 
    def shift_multiple_bond(self,rand):
        '''shift_multiple_bond() will generate a new array new_J.'''
        active_J_index = []
        self.new_J = copy.deepcopy(self.J)
        for i in range(rand,self.L):
            n2,n1 = randrange(self.N),randrange(self.N)
            x = np.random.normal(loc=0,scale=1.0,size=1) 
            # scale denotes standard deviation; 
            # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
            self.new_J[i][n2][n1] = (self.J[i][n2][n1] + x * rat) / RESCALE_J
            active_J_index.append((i,n2,n1))
            # record this step
            self.count_J_update += 1
        # change the active-or-not state of J
        self.J_active = True 
        return active_J_index
    def accept_by_mu_l_n(self,mu,l,n):
        """This accept function is used if S is flipped."""
        self.S[l,mu,n] = self.new_S[l,mu,n]
        #self.J = self.new_J
        self.H = self.H + self.delta_H
        print("ENERGY: {}".format(self.H))
    def accept_by_l_n2_n1(self,l,n2,n1):
        """This accept function is used if J is shifted."""
        self.J[l,n2,n1] = self.new_J[l,n2,n1]
        self.H = self.H + self.delta_H
        print("ENERGY: {}".format(self.H))
    def accept_by_mu_l_n_multiple(self,active_S_index):
        """This accept function is used if S is flipped."""
        for term in active_S_index:
            self.S[term[0],term[1],term[2]] = self.new_S[term[0],term[1],term[2]]
            #self.J = self.new_J
        self.H = self.H + self.delta_H
        print("ENERGY: {}".format(self.H))
    def accept_by_l_n2_n1_multiple(self,active_J_index):
        """This accept function is used if mulpile J is shifted."""
        for term in active_J_index:
            self.J[term[0],term[1],term[2]] = self.new_J[term[0],term[1],term[2]]
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
        part_gap[0] = (np.sum( self.J[l-1,n,:] * self.S[l-1,mu,:])/SQRT_N) * self.S[l,mu,n] 
        for n2 in range(self.N):
            part_gap[1+n2] = (np.sum(self.J[l,n2,:] * self.S[l,mu,:])/SQRT_N) * self.S[l+1,mu,n2] 
        return part_gap  # Only the N+1 elements affect the Delta_H_eff. 
    def part_gap_after_flip(self,mu,l,n):
        '''Ref: Yoshino2019, eqn (31b)
           When S is fliped, only one machine changes its coordinates and it will affect the gap of the node before it and the gaps of the N nodes
           behind it. Therefore, N+1 gaps contributes to the Delta_H_eff. l = 0,1, ..., L-1. 
           We define a small array, part_gap, which has N+1 elements. Each elements of part_gap is a r^mu_node. Use part_gap, we can calculate the 
           Energy change coused by the flip of S^mu_node,n. 
        '''
        part_gap = np.zeros(self.N + 1)
        part_gap[0] = (np.sum( self.J[l-1,n,:] * self.S[l-1,mu,:])/SQRT_N) * self.new_S[l,mu,n] 
        for n2 in range( self.N):
            part_gap[1+n2] = (np.sum(self.J[l,n2,:] * self.new_S[l,mu,:])/SQRT_N) * self.S[l+1,mu,n2] 
        return part_gap  # Only the N+1 elements affect the Delta_H_eff. 
    def part_gap_before_shift(self,l,n): 
        '''Ref: Yoshino2019, eqn (31b)
           When J is fliped, all M machine change their coordinates and they will affect the gap of the nodes before them.
           Therefore, 1 * M gaps contributes to the Delta_H_eff. mu = 0,1, ..., M-1; l=l;n=n1. 
           We define a small array, part_gap, which has M elements. Each elements of part_gap is a r^mu_node. Use part_gap, we can calculate the 
           Energy change coused by the shift of of J_node,n. 
        '''
        part_gap = np.zeros(self.M)
        for mu in range(self.M):
            part_gap[mu] = (np.sum(self.J[l,n,:] * self.S[l,mu,:])/SQRT_N) * self.S[l+1,mu,n] 
        return part_gap  # Only the M elements affect the Delta_H_eff. 
    def part_gap_after_shift(self,l,n): 
        '''Ref: Yoshino2019, eqn (31b)
           When J is fliped, all M machine change their coordinates and they will affect the gap of the nodes before them.
           Therefore, 1 * M gaps contributes to the Delta_H_eff. mu = 0,1, ..., M-1; l=l;n=n1. 
           We define a small array, part_gap, which has M elements. Each elements of part_gap is a r^mu_node. Use part_gap, we can calculate the 
           Energy change coused by the shift of of J_node,n. 
        '''
        part_gap = np.zeros(self.M)
        for mu in range(self.M):
            part_gap[mu] = (np.sum(self.new_J[l,n,:] * self.S[l,mu,:])/SQRT_N) * self.S[l+1,mu,n] 
        return part_gap # Only the M elements affect the Delta_H_eff. 
    def part_gap_before_flip_multiple(self,active_S_index):
        part_gap = []
        for term in active_S_index:
            part_gap.append(  (np.sum( self.J[term[0]-1,term[2],:] * self.S[term[0]-1,term[1],:])/SQRT_N) * self.S[term[0],term[1],term[2]] ) 
            for n2 in range(self.N):
                part_gap.append( (np.sum(self.J[term[0],n2,:] * self.S[term[0],term[1],:])/SQRT_N) * self.S[term[0]+1,term[1],n2] ) 
        return np.array(part_gap)  
    def part_gap_after_flip_multiple(self,active_S_index):
        part_gap = []
        for term in active_S_index:
            part_gap.append( (np.sum( self.J[term[0]-1,term[2],:] * self.S[term[0]-1,term[1],:])/SQRT_N) * self.new_S[term[0],term[1],term[2]] ) 
            for n2 in range(self.N):
                part_gap.append( (np.sum(self.J[term[0],n2,:] * self.new_S[term[0],term[1],:])/SQRT_N) * self.S[term[0]+1,term[1],n2] ) 
        return np.array(part_gap)  
    def part_gap_before_shift_multiple(self,active_J_index): 
        part_gap = np.zeros((self.M,len(active_J_index))) 
        for mu in range(self.M):
            for index,term in enumerate(active_J_index):
                part_gap[mu][index] = (np.sum(self.J[term[0],term[1],:] * self.S[term[0],mu,:])/SQRT_N) * self.S[term[0]+1,mu,term[1]] 
        return part_gap
    def part_gap_after_shift_multiple(self,active_J_index): 
        part_gap = np.zeros((self.M,len(active_J_index))) 
        for mu in range(self.M):
            for index,term in enumerate(active_J_index):
                part_gap[mu][index] = (np.sum(self.new_J[term[0],term[1],:] * self.S[term[0],mu,:])/SQRT_N) * self.S[term[0]+1,mu,term[1]]
        return part_gap  
    def decision_by_mu_l_n(self,MC_index,mu,l,n):
        """
        1. np.random.random(1) generate a random float number between 0 and 1.
        3. k_B = 1.
        """
        self.delta_H = calc_ener(self.part_gap_after_flip(mu,l,n)) - calc_ener(self.part_gap_before_flip(mu,l,n))
        delta_e = self.delta_H
        #print("[S]delta_E:{}".format(delta_e)) 
        if delta_e < 0:
            # replace o.S by o.new_S: use copy.deepcopy() 
            self.accept_by_mu_l_n(mu,l,n) 
            print("ACCEPT.")       
        else:
            if np.random.random(1) < np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n(mu,l,n)
                print("ACCEPT.")       
            else:
                pass
                #print("REJECTED.")       
    def decision_by_l_n2_n1(self,MC_index,l,n2,n1):
        """
        1. np.random.random(1) generate a random float number between 0 and 1.
        3. k_B = 1.
        """
        self.delta_H = calc_ener(self.part_gap_after_shift(l,n2)) - calc_ener(self.part_gap_before_shift(l,n2))
        delta_e = self.delta_H
        #print("[J]delta_E:{}".format(delta_e)) 
        if delta_e < 0:
            # replace o.S by o.new_S: use copy.deepcopy() 
            self.accept_by_l_n2_n1(l,n2,n1) 
            print("ACCEPT.")       
        else:
            if np.random.random(1) < np.exp(-delta_e * self.beta):
                self.accept_by_l_n2_n1(l,n2,n1)
                print("ACCEPT.")       
            else:
                pass
                #print("REJECTED.")       
    def decision_by_mu_l_n_multiple(self,MC_index,active_S_index):
        """
        1. np.random.random(1) generate a random float number between 0 and 1.
        3. k_B = 1.
        """
        self.delta_H = calc_ener(self.part_gap_after_flip_multiple(active_S_index)) - calc_ener(self.part_gap_before_flip_multiple(active_S_index))
        delta_e = self.delta_H
        print("[S]delta_E:{}".format(delta_e)) 
        if delta_e < 0:
            # replace o.S by o.new_S: use copy.deepcopy() 
            self.accept_by_mu_l_n_multiple(active_S_index) 
            print("ACCEPT.")       
        else:
            if np.random.random(1) < np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n_multiple(active_S_index)
                print("ACCEPT.")       
            else:
                pass
                #print("REJECTED.")       
    def decision_by_l_n2_n1_multiple(self,MC_index,active_J_index):
        """
        1. np.random.random(1) generate a random float number between 0 and 1.
        3. k_B = 1.
        """
        self.delta_H = calc_ener(self.part_gap_after_shift_multiple(active_J_index)) - calc_ener(self.part_gap_before_shift_multiple(active_J_index))
        delta_e = self.delta_H
        #print("[J]delta_E:{}".format(delta_e)) 
        if delta_e < 0:
            # replace o.S by o.new_S: use copy.deepcopy() 
            self.accept_by_l_n2_n1_multiple(active_J_index) 
            print("ACCEPT.")       
        else:
            if np.random.random(1) < np.exp(-delta_e * self.beta):
                self.accept_by_l_n2_n1_multiple(active_J_index)
                print("ACCEPT.")       
            else:
                pass
                #print("REJECTED.")       

if __name__=='__main__':
    #Parameters for rescaling J
    rat = 0.1 # r:Yoshino2019 Eq(35)
    RESCALE_J = np.sqrt(1+rat**2)
    #If the update multiple S or J? 
    multiple_update = True 
    # Initialization
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
    #extra_index = 1 # A parameter used for helping plotting figures in log scale
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
    o = host_network(N,M,L,tot_steps,beta,start_timestamp)
    # define some parameters
    SQRT_N = np.sqrt(o.N)
    num_nodes = int(o.N*o.M*o.L)
    num_variables = int(o.N*o.M*(o.L-1))
    num_bonds = int(o.N*o.N*o.L)
    num = num_variables+num_bonds
    ratio_for_sites = num_variables/num
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
    print("ENERGY (first step):{}".format(o.H_traj[1]))

    # MC 
    # For save J and S sequences
    name_S_seq = '{}/{}/seq_S_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.csv'.format(data,start_timestamp,L,M,N,beta,tot_steps)
    name_J_seq = '{}/{}/seq_J_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.csv'.format(data,start_timestamp,L,M,N,beta,tot_steps)
    file_o_S_seq = open(name_S_seq, 'w')
    file_o_J_seq = open(name_J_seq, 'w')
    # MC siulation starts
    if multiple_update:
        reduced_tot_steps = int(tot_steps/(L/2))
        for MC_index in range(1,reduced_tot_steps):
            print("MC step:{:d}".format(MC_index))
            for update_index in range(num):
                if np.random.random(1) < ratio_for_sites:
                    #print("  FOR FLIPPINT S")
                    active_S_index = o.flip_multiple_S(choice([0,1]))
                    o.decision_by_mu_l_n_multiple(MC_index,active_S_index)
                else:
                    #print("  FOR SHIFTING J")
                    active_J_index = o.shift_multiple_bond(choice([0,1])) 
                    o.decision_by_l_n2_n1_multiple(MC_index,active_J_index)
            o.count_MC_step += 1
            o.S_traj[MC_index] = o.S  #
            o.J_traj[MC_index] = o.J
            o.H_traj[MC_index] = o.H
    else:
        for MC_index in range(1,tot_steps):
            print("MC step:{:d}".format(MC_index))
            for update_index in range(num):
                if np.random.random(1) < ratio_for_sites:
                    #print("  FOR FLIPPINT S")
                    o.flip_S()
                    o.decision_by_mu_l_n(MC_index,o.updating_sample_index,o.updating_layer_index, o.updating_node_index)
                else:
                    #print("  FOR SHIFTING J")
                    o.shift_bond() 
                    o.decision_by_l_n2_n1(MC_index,o.updating_layer_index, o.updating_node_index_n2,o.updating_node_index_n1)
            o.count_MC_step += 1
            o.S_traj[MC_index] = o.S  #
            o.J_traj[MC_index] = o.J
            o.H_traj[MC_index] = o.H
    # MC is done, we can close some files for recording the S and J sequences.
    file_o_S_seq.close()
    file_o_J_seq.close()
    np.save('{}/{}/S_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data,start_timestamp,L,M,N,beta,tot_steps),o.S_traj)
    np.save('{}/{}/J_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data,start_timestamp,L,N,beta,tot_steps),o.J_traj)
    np.save('{}/{}/ener_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data,start_timestamp,L,M,N,beta,tot_steps),o.H_traj)
    print("All MC simulations done!")
    print('Time taken to run: {:5.1f} seconds.'.format(int(time())-start_time_int))
