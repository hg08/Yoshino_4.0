#======
#Module
#======
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
#import shutil # for moving files

class guest_network:
    def __init__(self,timestamp):
        #=====================================================================================
        # S and J are the coordinates of the machine.
        # This is a quest machine, so the intial configurations are from the teacher machine. 
        # In teacher machine, we intialize the S and J in the following way:
        # S = init_S(M,L,N)
        # J = init_J(L,N)
        #-------------------------------------------------------------------------------------
        # the argument timestamp come from outside of the function.
        # The reason we need timestamp: the initial configurations of S and J in the guest 
        # machine should be the same. The teacher machine generated the configuration it used
        # in a MC simulation, and saved them into a directory, which is named by HG with the 
        # timestamp of the moment at which the teacher.py start runing. 
        #=====================================================================================
        """In this new class (in Yoshino_3.0), when we update the energy, we do not calculate all the gaps, but only calculate these affected by the flip of a SPIN (S)  or a shift of 
           a BOND (J). This will accelerate the calculation by hundreds of times. (2) Besides, we also note that we do NOT need to define a functon: remain(), which will record the 
           new MC steps' S, J and H, even though one MC move is rejected."""
        # to obtain the basic integer parameters: M, L, N, tot_steps
        para_list = np.load('../data/{:s}/para_list_basic.npy'.format(timestamp))
        beta_tmp = np.load('../data/{:s}/para_list_beta.npy'.format(timestamp))
        L = para_list[0]
        M = para_list[1]
        N = para_list[2]
        tot_steps = para_list[-1]
        # to obtain the float parameter: inverse temperature (beta)
        beta = beta_tmp[0]
        # then store these parameters in the guest machine
        self.L = L 
        self.M = M 
        self.N = N 
        self.tot_steps = tot_steps
        self.beta = beta
        
        # Define new parameters; T (technically required)
        self.T = int(tot_steps+1)  # we keep the initial state in the first step 
        self.timestamp = timestamp  

        # The arrays for storing MC trajectories of S, J; the Accept/Reject probability of S and J; H
        self.S_traj = np.zeros((self.T, self.L+1, self.M, self.N))
        self.J_traj = np.zeros((self.T, self.L, self.N, self.N)) 
        self.H_traj = np.zeros(self.T)
        
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
        self.S = np.load('../data/{:s}/seed_S_L{:d}_M{:d}_N{:d}_beta{:3.1f}.npy'.format(timestamp,L,M,N,beta))
        self.J = np.load('../data/{:s}/seed_J_L{:d}_N{:d}_beta{:3.1f}.npy'.format(timestamp,L,N,beta))
        self.new_S = copy.deepcopy(self.S) # for storing temperay array when update 
        self.new_J = copy.deepcopy(self.J) # for storing temperay array when update  
        #self.new_S = self.S # for storing temperay array when update 
        #self.new_J = self.J # for storing temperay array when update  
        self.r = self.gap_init() # the initial gap is returned from a function.

        #self.new_r = self.r
        self.new_r = copy.deepcopy(self.r)

        self.H = 0 # for storing energy when update
        self.new_H = 0 # for storing temperay energy when update

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
        #self.new_S = self.S 
        self.new_S = copy.deepcopy(self.S)
        i,j,k = randrange(1,self.L), randrange(self.M), randrange(self.N)
        # update self.new_S
        #self.new_S[1+i][j][k] = -1 * self.new_S[1+i][j][k]  
        self.new_S[i][j][k] = -1 * self.new_S[i][j][k]  
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
        #x = np.random.normal(loc=0,scale=1.0,size=None) 
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
        self.S[l,mu,n] = self.new_S[l,mu,n]
        #self.J = self.new_J
        self.H = self.H + self.delta_H
        print("ENERGY: {}".format(self.H))

    def accept_by_l_n2_n1(self,l,n2,n1):
        self.J[l,n2,n1] = self.new_J[l,n2,n1]
        #self.J = self.new_J
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
        for n2 in range(self.N):
            #self.r[mu,l,n] = np.sum(self.new_J[l,n,:] * self.new_S[l,mu,:] * self.new_S[l+1,mu,n]/np.sqrt(self.N)).sum()
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
        return part_gap  # Only the M elements affect the Delta_H_eff. 

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
            part_gap.append(  (np.sum( self.J[term[0]-1,term[2],:] * self.S[term[0]-1,term[1],:])/SQRT_N) * self.new_S[term[0],term[1],term[2]] ) 
            for n2 in range(self.N):
                part_gap.append( (np.sum(self.J[term[0],n2,:] * self.new_S[term[0],term[1],:])/SQRT_N) * self.S[term[0]+1,term[1],n2] ) 
        return np.array(part_gap)  
    def part_gap_before_shift_multiple(self,active_J_index): 
        part_gap = [] 
        for mu in range(self.M):
            for term in active_J_index:
                part_gap.append( (np.sum(self.J[term[0],term[1],:] * self.S[term[0],mu,:])/SQRT_N) * self.S[term[0]+1,mu,term[1]] ) 
        return np.array(part_gap)  
    def part_gap_after_shift_multiple(self,active_J_index): 
        part_gap = [] 
        for mu in range(self.M):
            for term in active_J_index:
                part_gap.append((np.sum(self.new_J[term[0],term[1],:] * self.S[term[0],mu,:])/SQRT_N) * self.S[term[0]+1,mu,term[1]]) 
        return np.array(part_gap)  
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
    def decision_by_mu_l_n(self,MC_index,mu,l,n):
        """
        1. np.random.random(1) generate a random float number between 0 and 1.
        3. k_B = 1.
        """
        self.delta_H = calc_ener(self.part_gap_after_flip(mu,l,n)) - calc_ener(self.part_gap_before_flip(mu,l,n))
        delta_e = self.delta_H
        if delta_e < 0:
            # replace o.S by o.new_S: use copy.deepcopy() 
            self.accept_by_mu_l_n(mu,l,n) 
            print("ACCEPT.")       
        else:
            if np.random.random(1)< np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n(mu,l,n)
                print("ACCEPT.")       
            else:
                pass
                #self.remain(MC_index)
    def decision_by_l_n2_n1(self,MC_index,l,n2,n1):
        """
        1. np.random.random(1) generate a random float number between 0 and 1.
        3. k_B = 1.
        """
        self.delta_H = calc_ener(self.part_gap_after_shift(l,n2)) - calc_ener(self.part_gap_before_shift(l,n2))
        delta_e = self.delta_H
        if delta_e < 0:
            # replace o.S by o.new_S: use copy.deepcopy() 
            self.accept_by_l_n2_n1(l,n2,n1) 
            print("ACCEPT.")       
        else:
            if np.random.random(1)< np.exp(-delta_e * self.beta):
                self.accept_by_l_n2_n1(l,n2,n1)
                print("ACCEPT.")       
            else:
                pass # We do not need remain function
                #self.remain(MC_index)
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
    #def remain(self,MC_index):
    #    """
    #    Why we need remain()? A: If a move is rejected, we will think there is also a step, we also have to record the traj of S, J and H.
    #    Note:MC_index > 0 """
    #    self.S_traj[MC_index] = self.S_traj[MC_index-1]
    #    self.J_traj[MC_index] = self.J_traj[MC_index-1] 
    #    self.H_traj[MC_index] = self.H_traj[MC_index-1]

# =====
# Main
# =====
if __name__=='__main__':
    #Rescaling J
    #parameters
    rat = 0.1 # r: Yoshino2019 Eq(35)
    RESCALE_J = np.sqrt(1+rat**2)
    #If the update multiple S or J? 
    multiple_update = True 
    # Initialization
    MC_index = 0

    # Obtain the timestamp list
    # Startint time
    start = time()
    data_dir = '../data'
    timestamp_list = list_only_naked_dir(data_dir)
    #print(str2int(data_list))
    #timestamp_list = str2int(data_list)

    import argparse
    #extra_index = 1 # A parameter used for helping plotting figures in log scale
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
    num_nodes = int(o.N*o.M*o.L)
    num_variables = int(o.N*o.M*(o.L-1))
    num_bonds = int(o.N*o.N*o.L)
    num = num_variables+num_bonds
    ratio_for_sites = num_variables/num

    o.S_traj[0,:,:,:] = o.S # Note that self.S_traj will independent of self.S from now on.
    o.J_traj[0,:,:,:] = o.J 
    o.H = calc_ener(o.r) # the energy
    o.H_traj[1] = o.H # H_traj[0] will be neglected
    print("ENergy: first step:")
    print(o.H)
    print(o.H_traj[1])
    tot_steps = o.tot_steps

    # MC 
    # For save J and S sequences
    name_S_seq = '../data/{}/seq_S_guest_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.csv'.format(timestamp,o.L,o.M,o.N,o.beta,tot_steps)
    name_J_seq = '../data/{}/seq_J_guest_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.csv'.format(timestamp,o.L,o.M,o.N,o.beta,tot_steps)
    file_o_S_seq = open(name_S_seq, 'w')
    file_o_J_seq = open(name_J_seq, 'w')
    # MC siulation starts
    if multiple_update:
        tot_steps = int(tot_steps/(o.L/2))
        for MC_index in range(1,tot_steps):
            print("MC step:{:d}".format(MC_index))
            for update_index in range(num):
                if np.random.random(1) < ratio_for_sites:
                    # Flip one spin and make a decision: there are M*(L-1)*N times
                    print("  FOR FLIPPINT S")
                    active_S_index = o.flip_multiple_S(choice([0,1]))
                    o.decision_by_mu_l_n_multiple(MC_index,active_S_index)
                else:
                    print("  FOR SHIFTING J")
                    # shift one bond (interaction) and make a decision: there are L*N*N times
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
                    # Flip one spin and make a decision: there are M*(L-1)*N times
                    print("  FOR FLIPPINT S")
                    #o.flip_multiple_S(choice([0,1]))
                    o.flip_S()
                    o.decision_by_mu_l_n(MC_index,o.updating_sample_index,o.updating_layer_index, o.updating_node_index)
                else:
                    print("  FOR SHIFTING J")
                    # shift one bond (interaction) and make a decision: there are L*N*N times
                    o.shift_bond() 
                    o.decision_by_l_n2_n1(MC_index,o.updating_layer_index, o.updating_node_index_n2,o.updating_node_index_n1)
            o.count_MC_step += 1
            o.S_traj[MC_index] = o.S  #
            o.J_traj[MC_index] = o.J
            o.H_traj[MC_index] = o.H
    # MC is done, we can close some files for recording the S and J sequences.
    file_o_S_seq.close()
    file_o_J_seq.close()
    
    np.save('../data/{:s}/S_guest_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(timestamp,o.L,o.M,o.N,o.beta,tot_steps),o.S_traj)
    np.save('../data/{:s}/J_guest_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(timestamp,o.L,o.N,o.beta,tot_steps),o.J_traj)
    np.save('../data/{:s}/ener_guest_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(timestamp,o.L,o.M,o.N,o.beta,tot_steps),o.H_traj)
    
    # Finished
    print("MC simulations are done.")
    print(f'Time taken to run: {time() - start} seconds')
