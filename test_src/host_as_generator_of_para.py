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
import copy
from utilities_old import *
from time import time

#=========
#The Class
#=========
class host_network:
    def __init__(self,L,M,N,tot_steps,beta,timestamp):
        # store the parameters used in the host machine
        self.L = int(L) # number of layers
        self.M = int(M) # number of samples
        self.N = int(N) # the number of neurons at each layer.
        self.tot_steps = int(tot_steps) 
        self.timestamp = timestamp  
        self.beta = beta # inverse temperature
        
        # Define new parameters; T (technically required)
        T = self.tot_steps + 1  # we keep the initial state in the first step 

        # The arrays for storing MC trajectories of S, J; the Accept/Reject probability of S and J; H
        self.S_traj = np.zeros((T, self.M, self.L, self.N))
        self.J_traj = np.zeros((T, self.L, self.N, self.N))
        self.H_traj = np.zeros(T)
        
        # For recording which kind of coordinate (S or J) is activated
        self.S_active = False 
        self.J_active = False 
        #Intialize S and J by the array S and J. 
        #Note: Both S and J are the coordinates of a machine.
        self.S = init_S(self.M, self.L, self.N)
        self.J = init_J(self.L, self.N) 
        self.new_S = self.S # for storing temperay array when update 
        self.new_J = self.J # for storing temperay array when update  
        self.r = self.gap_init() # the initial gap is returned from a function.
        self.H = 0 # for storing energy when update
        self.new_H = 0 # for storing temperay energy when update

        # Initialize the simulation steps
        self.count_S_update = 0           
        self.count_J_update = 0           
        self.count_MC_step = 0           

    def flip_S(self,mu,l,n):
        '''flip_S() will generate a new array new_S. Note: all the spins can be update except the input/output.'''
        # record this step
        self.new_S = copy.deepcopy(self.S)
        # self.new_S = self.S 
        #i,j,k = randrange(self.M), randrange(1,self.L), randrange(self.N)
        # update self.new_S
        self.new_S[mu][l][n] = -1 * self.new_S[mu][l][n]  
        # record this step
        self.count_S_update += 1          
        # change the active-or-not state of S 
        self.S_active = True 

    def shift_bond(self,l,n2,n1,x):
        '''shift_bond() will generate a new array new_J.'''
        self.new_J = copy.deepcopy(self.J)
        # Do not introduce randomness in a function.
        # scale denotes standard deviation; 
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J[l][n2][n1] = (self.J[l][n2][n1] + x * rat) / RESCALE_J
        # record this step
        self.count_J_update += 1
        # change the active-or-not state of J
        self.J_active = True 
    
    def accept(self):
        self.S = copy.deepcopy(self.new_S)
        self.J = copy.deepcopy(self.new_J)
        self.H = copy.deepcopy(self.new_H) 
        print("ENERGY: {}".format(self.H))

    def gap_init(self):
        '''Ref: Yoshino2019, eqn (31b)'''
        r = np.ones((self.M,self.L,self.N))
        for mu in range(self.M):
            for l in range(1,self.L): # l = 0,...,L-1
                for n in range(self.N):
                    #r[mu,l,n] = np.sum(self.J[l,n,:] * self.S[l,mu,:] * self.S[l+1,mu,n]/np.sqrt(self.N)).sum()
                    r[mu,l,n] = (np.sum(self.J[l,n,:] * self.S[mu,l-1,:])/np.sqrt(self.N)) * self.S[mu,l,n] 
        return r    
    def gap(self):
        '''Ref: Yoshino2019, eqn (31b)'''
        #r = np.ones((self.M,self.L,self.N))
        for mu in range(self.M):
            for l in range(1,self.L):
                for n in range(self.N):
                    #self.r[mu,l,n] = np.sum(self.new_J[l,n,:] * self.new_S[l,mu,:] * self.new_S[l+1,mu,n]/np.sqrt(self.N)).sum()
                    self.r[mu,l,n] = (np.sum(self.new_J[l,n,:] * self.new_S[mu,l-1,:])/np.sqrt(self.N)) * self.new_S[mu,l,n] 
    def decision(self,MC_index,rand):
        """
        1. np.random.random(1) generate a random float number between 0 and 1.
        2. accept probability=min(1, exp(-\beta \Delta_E))
        3. k_B = 1.
        """
        #r = self.gap()
        self.gap()
        self.new_H = calc_ener(self.r)
        delta_e = self.new_H - self.H
        if delta_e < 0:
            # replace o.S by o.new_S: use copy.deepcopy() 
            self.accept() 
            #print("ACCEPT.")       
        else:
            #if np.random.random(1)< np.exp(-delta_e * self.beta):
            if rand < np.exp(-delta_e * self.beta):
                self.accept()
                #print("ACCEPT.")       
            else:
                pass

    def rand_index_for_S(self):
        # For S: list_index_for_S = [(mu,l,n),...]
        list_index_for_S = []
        for i in range(num_variables * (self.tot_steps-1)):
            list_index_for_S.append([randrange(self.M), randrange(1,self.L-1), randrange(self.N)])
        res_arr = np.array(list_index_for_S)
        return res_arr
    def rand_index_for_J(self):
        # For generating J: list_index_for_J = [(l,n2,n1),...]
        list_index_for_J = []
        for i in range(num_bonds * (self.tot_steps-1)):
            list_index_for_J.append([randrange(1,self.L), randrange(self.N), randrange(self.N)])
        res_arr = np.array(list_index_for_J)
        return res_arr
    def rand_series_for_x(self):
        """ 
        For generating J: list_for_x = [x1,x2,...]
        We separate rand_index_for_J() and rand_series_for_x(), instead of merginging them to one function and return a list of four-tuple (l,n2,n1,x).
        The reason is: x is float and l,n2,n1 are integers, it will induce trouble if one put them (x and l,n2,n1 ) together.
        """
        list_for_x = []
        for i in range(num_bonds * (self.tot_steps-1)):
            x = np.random.normal(loc=0,scale=1.0,size=None)
            list_for_x.append(x)
        res_arr = np.array(list_for_x)
        return res_arr

    def rand_series_for_decision_on_S(self):
        # For generating J: list_index_for_J = [(l,n2,n1),...]
        list_for_decision = []
        for i in range(num_variables * (self.tot_steps-1)):
            list_for_decision.append(np.random.random(1))
        res_arr = np.array(list_for_decision)
        return res_arr
    def rand_series_for_decision_on_J(self):
        # For generating J: list_index_for_J = [(l,n2,n1),...]
        list_for_decision = []
        for i in range(num_bonds * (self.tot_steps-1)):
            list_for_decision.append(np.random.random(1))
        res_arr = np.array(list_for_decision)
        return res_arr

if __name__=='__main__':
    rat = 0.1 # r: Yoshino2019 Eq(35)
    RESCALE_J = np.sqrt(1+rat**2)
    MC_index = 0
    #=================================================================================
    # start_timestamp has two functions: 1st, calculate the time consumed by the code;
    # 2nd, used as the name of directory where the parameters will be located.
    #=================================================================================
    start = time()
    start_timestamp = int(time())
    print("starting time:{}".format(int(start_timestamp)))
    # Read the arguments
    import argparse
    #extra_index = 1 # A parameter used for helping plotting figures in log scale
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', nargs='?', const=50, type=int, default=50, \
                        help="the number of samples.")
    parser.add_argument('-L', nargs='?', const=10, type=int, default=10, \
                        help="the number of hidden layers.(Condition: L > 1)") 
    parser.add_argument('-N', nargs='?', const=5, type=int, default=5, \
                        help="the number of nodes per layer.") 
    parser.add_argument('-S', nargs='?', const=10, type=int, default=10, \
                        help="the number of total steps.")
    parser.add_argument('-B', nargs='?', const=66.7, type=float, default=66.7, \
                        help="the inverse temperautre.") 
    args = parser.parse_args()
    M,N,L,beta,tot_steps = args.M, args.N, args.L, args.B, args.S

    # Preparing parameters for using in student machine
    parameter_list_basic = np.array([L,M,N,tot_steps])
    parameter_list_beta = np.array([beta])

    # Initilize an instance of network.
    o = host_network(L,M,N,tot_steps,beta,start_timestamp)
    # define some parameters
    num_nodes = int(o.N*o.M*o.L)
    num_variables = int(o.N*o.M*(o.L-1))
    num_bonds = int(o.N*o.N*o.L)
    num = num_variables+num_bonds

    # Generate two random list of 3-tuple for host machine: 
    # For S: list_index_for_S = [(mu,l,n),...]
    index_for_S =o.rand_index_for_S()
    # For J: list_index_for_J = [(l,n2,n1),...]
    index_for_J =o.rand_index_for_J()
    # For J: list_for_x = [x1,x2,...]
    series_for_x =o.rand_series_for_x()
    # For decision: list_for_decision = [p1,p2,...]
    series_for_decision_on_S =o.rand_series_for_decision_on_S()
    # For decision: list_for_decision = [p1,p2,...]
    series_for_decision_on_J =o.rand_series_for_decision_on_J()
    # Generate two random list of 3-tuple for guest machine: 
    # For S: list_index_for_S = [(mu,l,n),...]
    index_for_S_guest = o.rand_index_for_S()
    # For J: list_index_for_J = [(l,n2,n1),...]
    index_for_J_guest = o.rand_index_for_J()
    # For J: list_for_x = [x1,x2,...]
    series_for_x_guest = o.rand_series_for_x()
    # For decision: list_for_decision = [p1,p2,...]
    series_for_decision_on_S_guest =o.rand_series_for_decision_on_S()
    # For decision: list_for_decision = [p1,p2,...]
    series_for_decision_on_J_guest =o.rand_series_for_decision_on_J()
    #===================================
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
    np.save('{}/{:s}/index_for_S_L{:d}_M{:d}_N{:d}.npy'.format(data,start_timestamp,L,M,N),index_for_S)
    np.save('{}/{:s}/index_for_J_L{:d}_N{:d}.npy'.format(data,start_timestamp,L,N),index_for_J)
    np.save('{}/{:s}/series_for_x_L{:d}_N{:d}.npy'.format(data,start_timestamp,L,N),series_for_x)
    np.save('{}/{:s}/series_for_decision_on_S_L{:d}_N{:d}.npy'.format(data,start_timestamp,L,N),series_for_decision_on_S)
    np.save('{}/{:s}/series_for_decision_on_J_L{:d}_N{:d}.npy'.format(data,start_timestamp,L,N),series_for_decision_on_J)
    np.save('{}/{:s}/index_for_S_guest_L{:d}_M{:d}_N{:d}.npy'.format(data,start_timestamp,L,M,N),index_for_S_guest)
    np.save('{}/{:s}/index_for_J_guest_L{:d}_N{:d}.npy'.format(data,start_timestamp,L,N),index_for_J_guest)
    np.save('{}/{:s}/series_for_x_guest_L{:d}_N{:d}.npy'.format(data,start_timestamp,L,N),series_for_x_guest)
    np.save('{}/{:s}/series_for_decision_on_S_guest_L{:d}_N{:d}.npy'.format(data,start_timestamp,L,N),series_for_decision_on_S_guest)
    np.save('{}/{:s}/series_for_decision_on_J_guest_L{:d}_N{:d}.npy'.format(data,start_timestamp,L,N),series_for_decision_on_J_guest)
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
    for MC_index in range(1,tot_steps):
        print("MC step:{:d}".format(MC_index))
        for update_index in range( (MC_index-1)*num_variables,MC_index*num_variables ):
            # Flip one spin and make a decision: there are M*(L-1)*N times
            o.flip_S(index_for_S[update_index][0],index_for_S[update_index][1],index_for_S[update_index][2])
            o.decision(MC_index,series_for_decision_on_S[update_index])
            #print("  FOR FLIPPINT S")

        for update_index in range((MC_index-1)*num_bonds, MC_index*num_bonds):
            # shift one bond (interaction) and make a decision: there are L*N*N times
            o.shift_bond(index_for_J[update_index][0],index_for_J[update_index][1],index_for_J[update_index][2],series_for_x[update_index]) 
            o.decision(MC_index,series_for_decision_on_J[update_index])
            #print("  FOR SHIFTING J")
        o.count_MC_step += 1
        o.S_traj[MC_index] = o.S #
        o.J_traj[MC_index] = o.J
        o.H_traj[MC_index] = o.H
    # MC is done, we can close some files for recording the S and J sequences.
    file_o_S_seq.close()
    file_o_J_seq.close()
    np.save('{}/{}/S_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data,start_timestamp,L,M,N,beta,tot_steps),o.S_traj)
    np.save('{}/{}/J_L{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data,start_timestamp,L,N,beta,tot_steps),o.J_traj)
    np.save('{}/{}/ener_L{:d}_M{:d}_N{:d}_beta{:3.1f}_step{:d}.npy'.format(data,start_timestamp,L,M,N,beta,tot_steps),o.H_traj)
    print("All Monte Carlo simulations done!")
    print(f'Time taken to run: {time() - start} seconds')
